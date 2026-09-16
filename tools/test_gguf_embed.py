#!/usr/bin/env python3
"""Re-embed check for tools/gguf-embed.py: embed SRC -> A, then re-embed A -> B.
B must carry exactly one baro.kernel.* set, every non-baro KV of SRC in order, and a
tensor-data region byte-identical to SRC's. Writes under .work/embed-test/ (never /tmp).
usage: tools/test_gguf_embed.py SRC.gguf   (a small GGUF is enough; data is copied verbatim)
"""
import hashlib, struct, subprocess, sys
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = spec_from_file_location("ge", ROOT / "tools" / "gguf-extract.py")
ge = module_from_spec(spec); spec.loader.exec_module(ge)


def keys(path):
    with open(path, "rb") as f:
        f.read(8); n_t, n_kv = struct.unpack("<QQ", f.read(16)); out = []
        for _ in range(n_kv):
            k = ge.read_str(f); k = k.decode() if isinstance(k, bytes) else k
            (vt,) = struct.unpack("<I", f.read(4)); ge.read_value(f, vt, want=False); out.append(k)
    return out


def data_hash(path):
    _, _, data_start, _ = ge.parse(path)
    h = hashlib.sha256(); n = 0
    with open(path, "rb") as f:
        f.seek(data_start)
        while chunk := f.read(1 << 24): h.update(chunk); n += len(chunk)
    return h.hexdigest()[:16], n


src = Path(sys.argv[1]); d = ROOT / ".work" / "embed-test"; d.mkdir(parents=True, exist_ok=True)
a, b = d / "A.gguf", d / "B.gguf"
for p in (a, b):
    p.unlink(missing_ok=True)
files = subprocess.run([sys.executable, str(ROOT / "tools/embed-files.py")], capture_output=True, text=True, cwd=ROOT).stdout.split()
for s_, d_ in ((src, a), (a, b)):
    subprocess.run([sys.executable, str(ROOT / "tools/gguf-embed.py"), str(s_), str(d_), *files], check=True, cwd=ROOT)
ks, kb = keys(src), keys(b)
base = [k for k in ks if not k.startswith("baro.kernel.")]
baro_b = [k for k in kb if k.startswith("baro.kernel.")]
ok = True
def check(name, cond, detail):
    global ok; ok &= cond; print(("PASS " if cond else "FAIL ") + name + ": " + detail)
check("one baro set", len(baro_b) == len(set(baro_b)) and kb.count("baro.kernel.commit") == 1, f"{len(baro_b)} baro keys, commit x{kb.count('baro.kernel.commit')}")
check("sources", sum(k.startswith("baro.kernel.src.") for k in kb) == len(files), f"{sum(k.startswith('baro.kernel.src.') for k in kb)} src keys for {len(files)} files")
check("non-baro kv preserved", [k for k in kb if not k.startswith(("baro.kernel.", "baro.pad"))] == base, f"{len(base)} keys in order")
check("one pad kv", kb.count("baro.pad") == 1, f"baro.pad x{kb.count('baro.pad')}")
(_, _, ds, _), (_, _, db, _) = ge.parse(src), ge.parse(b)
check("data aligned to src block offset", ds % 4096 == db % 4096, f"src {ds % 4096} dst {db % 4096} mod 4096")
(hs, ns), (hb, nb) = data_hash(src), data_hash(b)
check("tensor data non-empty", ns > 0, f"{ns} bytes (an empty region, e.g. shard 1 of a split, proves nothing)")
check("tensor data identical", (hs, ns) == (hb, nb), f"sha {hs} {ns} bytes vs {hb} {nb} bytes")
sys.exit(0 if ok else 1)
