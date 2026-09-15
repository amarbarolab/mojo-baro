#!/usr/bin/env python3
"""Embed baro kernel sources into a GGUF's metadata (self-describing model).

Writes a NEW file — never touches the source GGUF. Adds string KV pairs:
  baro.kernel.arch, baro.kernel.commit, baro.kernel.files (comma list),
  baro.kernel.src.<basename> = full source text
GGUF readers ignore unknown keys; tensor offsets are relative to the data
region so only the header padding changes.

Usage: tools/gguf-embed.py SRC.gguf DST.gguf $(tools/embed-files.py)
       (split layout, 2026-09-08: window.mojo + registry.mojo + kernel closure; never serve/engine.mojo)

Optional, for B1 (`tools/baro`), all in the separate baro.run.* namespace so
the closure walk above stays harness-free:
  --run-harness=serve/engine.mojo  the RUN-mode harness plus its sha256
  --run-prompt=FILE                token ids the closure decodes
  --run-ref=FILE                   the ids it must produce (GENERATED: line or bare)
  --run-pack-flags='--q4 ...'      how tools/engine-pack.py builds this model's pack
  --run-pack-tool=tools/engine-pack.py  the pack builder itself, so run mode needs no checkout
"""
import hashlib
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

ALIGN = 32


def w_str(out, s):
    b = s.encode("utf-8")
    out.write(struct.pack("<Q", len(b)))
    out.write(b)


def src_key(k):
    """Kernel/engine sources are keyed by basename (engine imports them flat);
    anything else keeps its repo-relative path so the closure can rebuild it."""
    root = Path(__file__).resolve().parent.parent
    if not k.resolve().is_relative_to(root):
        return f"{k.resolve().parent.name}/{k.name}"  # external package: <pkg>/<file>
    rel = k.resolve().relative_to(root).as_posix()
    return k.name if rel.startswith(("kernels/", "serve/")) else rel


def rewrite(src, dst, new_kv, drop=("baro.kernel.", "baro.hw.", "baro.run.")):
    """Write dst = src with every KV whose key starts with one of `drop`
    replaced by `new_kv` (a list of (key, string value) pairs), tensor infos
    and tensor data copied byte for byte.

    Factored out of main() so tools/gguf-receipt.py can extend one KV without
    a second copy of the container format. Returns (added, dropped).
    """
    f = open(src, "rb")
    magic, version = struct.unpack("<4sI", f.read(8))
    assert magic == b"GGUF" and version == 3
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    kv_start = f.tell()
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
    ge = module_from_spec(spec)
    spec.loader.exec_module(ge)
    _, infos, data_start, kv = ge.parse(src)
    f.seek(kv_start)
    keep = []
    for _ in range(n_kv):
        a = f.tell()
        key = ge.read_str(f)
        key = key.decode("utf-8") if isinstance(key, bytes) else key
        (vtype,) = struct.unpack("<I", f.read(4))
        ge.read_value(f, vtype, want=False)
        if not any(key.startswith(p) for p in drop):
            keep.append((a, f.tell()))
    kv_end = f.tell()
    for _ in range(n_tensors):
        ge.read_str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        f.seek(8 * nd + 4 + 8, 1)
    info_end = f.tell()

    out = open(dst, "wb")
    out.write(struct.pack("<4sI", b"GGUF", 3))
    out.write(struct.pack("<QQ", n_tensors, len(keep) + len(new_kv)))
    for a, b in keep:
        f.seek(a)
        out.write(f.read(b - a))
    for key, val in new_kv:
        w_str(out, key)
        out.write(struct.pack("<I", 8))
        w_str(out, val)
    f.seek(kv_end)
    out.write(f.read(info_end - kv_end))

    pad = (ALIGN - out.tell() % ALIGN) % ALIGN
    out.write(b"\x00" * pad)
    f.seek(data_start)
    while True:
        chunk = f.read(1 << 24)
        if not chunk:
            break
        out.write(chunk)
    out.close()
    return len(new_kv), n_kv - len(keep)


def main():
    extra = []
    args = []
    run_harness = None
    run_prompt = None
    run_ref = None
    run_pack_flags = None
    run_pack_tool = None
    for a in sys.argv[1:]:
        if a.startswith("--kv="):
            k, v = a[5:].split("=", 1)
            assert k.startswith("baro.") and (not k.startswith("baro.kernel.") or k == "baro.kernel.model"), f"--kv key must be baro.<ns>.<name> (or baro.kernel.model): {k}"
            extra.append((k, v))
        elif a.startswith("--run-harness="):
            run_harness = Path(a.split("=", 1)[1])
        elif a.startswith("--run-prompt="):
            run_prompt = Path(a.split("=", 1)[1])
        elif a.startswith("--run-ref="):
            run_ref = Path(a.split("=", 1)[1])
        elif a.startswith("--run-pack-flags="):
            run_pack_flags = a.split("=", 1)[1]
        elif a.startswith("--run-pack-tool="):
            run_pack_tool = Path(a.split("=", 1)[1])
        else:
            args.append(a)
    src, dst = Path(args[0]), Path(args[1])
    kfiles = [Path(p) for p in args[2:]]
    assert src.exists() and not dst.exists(), "dst must not exist"

    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent),
         "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    new_kv = [("baro.kernel.arch", "gfx1100"),
              ("baro.kernel.commit", commit),
              ("baro.kernel.files", ",".join(src_key(k) for k in kfiles))]
    for pkg in sorted({k.resolve().parent for k in kfiles if not k.resolve().is_relative_to(Path(__file__).resolve().parent.parent)}):
        c = subprocess.run(["git", "-C", str(pkg), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(pkg), "status", "--short"], capture_output=True, text=True).stdout.strip()
        assert not dirty, f"{pkg} dirty"
        new_kv.append((f"baro.kernel.ext.{pkg.name}.commit", c))
    if os.environ.get("BARO_KERNEL_PARENT"):
        new_kv.append(("baro.kernel.parent", os.environ["BARO_KERNEL_PARENT"]))
    # --kv=baro.hw.<name>=<value> (repeatable): the receipts a verifier on another
    # card compares against (card, driver, power cap, the 20-prompt number and the
    # protocol that produced it). Earlier baro.hw.* keys are replaced like the
    # kernel set, so a re-bake never carries two scoreboards.
    new_kv.extend(extra)
    for k in kfiles:
        new_kv.append((f"baro.kernel.src.{src_key(k)}", k.read_text()))

    # --run-harness=serve/engine.mojo: the RUN-mode harness (B1). The closure
    # walk stays harness-free -- baro.kernel.src.* never carries engine.mojo,
    # because a file that carries its own stopwatch cannot be timed by it
    # (exchange/scorer-integrity-report.md P-A). This is a separate namespace
    # for a separate job: `baro run` serves the model with no checkout and
    # labels every number it prints self-reported, while `baro verify` takes
    # the harness from git and first checks this copy is byte-identical to it,
    # which is how a tampered clock is caught.
    if run_harness is not None:
        text = run_harness.read_text()
        new_kv.append((f"baro.run.src.{run_harness.name}", text))
        new_kv.append(("baro.run.harness.sha", hashlib.sha256(text.encode()).hexdigest()))
        new_kv.append(("baro.run.harness.path", "serve/" + run_harness.name))
        # The harness's own serve-module closure (prefix, harness, moe_pack,
        # latent, ...), minus whatever the kernel closure already carries.
        # tools/gguf-closure.sh pulls these from git at the gguf's commit, which
        # verify mode can do and run mode cannot: run mode has no checkout, and
        # without them the build stops at "unable to locate module 'prefix'".
        serve_dir = run_harness.resolve().parent
        have = {src_key(k) for k in kfiles}
        seen, todo = set(), [run_harness]
        while todo:
            f = todo.pop(0)
            for m in re.findall(r"^(?:from|import)\s+([a-z_][a-z0-9_]*)", f.read_text(), re.M):
                cand = serve_dir / f"{m}.mojo"
                if m in seen or not cand.exists() or cand.name in have:
                    continue
                seen.add(m)
                todo.append(cand)
                new_kv.append((f"baro.run.src.{cand.name}", cand.read_text()))
        if seen:
            print(f"run closure: {len(seen)} extra serve module(s): {' '.join(sorted(seen))}")

    # The closure used to reach for .work/ paths that the file never carried
    # (the qwen35moe branch defaulted BARO_PROMPT to .work/moe-w3/one.tokens,
    # which no longer exists on this box, so tools/gguf-verify.sh exited 1 on
    # the MoE bake). A file that verifies itself carries its own prompt and its
    # own reference ids.
    if run_prompt is not None:
        new_kv.append(("baro.run.prompt.tokens", " ".join(run_prompt.read_text().split())))
    if run_ref is not None:
        ids = run_ref.read_text().replace("GENERATED:", " ").split()
        new_kv.append(("baro.run.ref.tokens", " ".join(ids)))
    if run_pack_flags is not None:
        new_kv.append(("baro.run.pack.flags", run_pack_flags))
    if run_pack_tool is not None:
        # The pack builder too, so run mode needs no checkout to turn the file's
        # own weights into the engine's pack.
        new_kv.append((f"baro.run.src.{run_pack_tool.name}", run_pack_tool.read_text()))

    # Existing KVs are copied except any earlier baro.kernel.*/baro.hw.*/baro.run.*
    # set: re-embedding from a BARO file replaces its sources and its scoreboard,
    # it never appends a second one.
    added, dropped = rewrite(src, dst, new_kv)
    print(f"wrote {dst} (+{added} kv, dropped {dropped} earlier baro.kernel.*/baro.hw.*/baro.run.* kv)")


if __name__ == "__main__":
    main()
