#!/usr/bin/env python3
"""Print the source files a BARO gguf must embed for the split engine layout:
serve/window.mojo, serve/registry.mojo and the transitive closure of the kernel
modules they import. serve/engine.mojo (pack load, stopwatch, prints) is NOT
embedded -- tools/gguf-closure.sh and tools/loop-gate.sh take it from git at the
gguf's own commit (exchange/scorer-integrity-report.md, P-A, 2026-09-08).

    tools/gguf-embed.py SRC.gguf DST.gguf $(tools/embed-files.py)
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def closure(roots):
    seen, todo = [], list(roots)
    while todo:
        f = todo.pop(0)
        if f in seen:
            continue
        seen.append(f)
        for m in re.finditer(r"^from (\w+) import|^import (\w+)", (ROOT / f).read_text(), re.M):
            mod = m.group(1) or m.group(2)
            for cand in (f"kernels/{mod}.mojo", f"serve/{mod}.mojo"):
                if (ROOT / cand).exists() and cand not in seen:
                    todo.append(cand)
    return sorted(seen)


if __name__ == "__main__":
    files = closure(["serve/window.mojo", "serve/registry.mojo"])
    assert "serve/engine.mojo" not in files
    print("\n".join(files) if "-1" in sys.argv else " ".join(files))
