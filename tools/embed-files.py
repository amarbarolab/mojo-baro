#!/usr/bin/env python3
"""Print the source files a BARO gguf must embed for the split engine layout:
serve/window.mojo, serve/registry.mojo and the transitive closure of the kernel
modules they import. serve/engine.mojo (pack load, stopwatch, prints) is NOT
embedded -- tools/gguf-closure.sh and tools/loop-gate.sh take it from git at the
gguf's own commit (exchange/scorer-integrity-report.md, P-A, 2026-09-08).

    tools/gguf-embed.py SRC.gguf DST.gguf $(tools/embed-files.py)
    tools/embed-files.py --arch spark   # serve/spark.mojo closure (spark.mojo itself is the
                                        # harness, taken from git); uregex/minja come in
                                        # whole, as a package glob (vendored at repo root
                                        # since M1, briefs/2026-09-15-wiring-lane.md -- no
                                        # longer external, but still not a single kernels/X.mojo
                                        # or serve/X.mojo candidate, so EXT is still how the
                                        # closure walker pulls a whole package directory in)
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = {"uregex": ROOT / "uregex", "minja": ROOT / "minja"}
ARCH = {"qwythos": (["serve/window.mojo", "serve/registry.mojo"], "serve/engine.mojo"),
        "spark": (["serve/spark.mojo"], "serve/spark.mojo"),
        # The W2 kernel-parity closure, kept for the expert-kernel gate.
        "qwen35moe-kernels": (["kernels/test_moe_block.mojo"], "kernels/test_moe_block.mojo"),
        # The real engine closure for the MoE profile: same roots as qwythos,
        # so the walk picks up kernels/moe.mojo, the three serve/model*.mojo
        # profiles AND kernels/mega.mojo. The MoE profile does not run the
        # megakernel (MEGA_ALLOWED is False for qwen35moe), but registry.mojo
        # instantiates it unconditionally, so it is part of the closure and the
        # file does not compile without it.
        "qwen35moe": (["serve/window.mojo", "serve/registry.mojo"], "serve/engine.mojo")}


def closure(roots):
    seen, ext, todo = [], [], list(roots)
    while todo:
        f = todo.pop(0)
        if f in seen:
            continue
        seen.append(f)
        text = (ROOT / f).read_text() if isinstance(f, str) else f.read_text()
        for m in re.finditer(r"^from (\w+)[.\w]* import|^import (\w+)", text, re.M):
            mod = m.group(1) or m.group(2)
            for cand in (f"kernels/{mod}.mojo", f"serve/{mod}.mojo"):
                if (ROOT / cand).exists() and cand not in seen:
                    todo.append(cand)
            if mod in EXT:
                for p in sorted(EXT[mod].glob("*.mojo")):
                    if p not in seen and p not in todo:
                        todo.append(p)
    return sorted(x for x in seen if isinstance(x, str)) + sorted(str(x) for x in seen if not isinstance(x, str))


if __name__ == "__main__":
    arch = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "qwythos"
    roots, harness = ARCH[arch]
    files = [f for f in closure(roots) if f != harness]
    assert "serve/engine.mojo" not in files
    print("\n".join(files) if "-1" in sys.argv else " ".join(files))
