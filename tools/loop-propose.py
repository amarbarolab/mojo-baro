#!/usr/bin/env python3
"""Proposer step of the self-optimising loop: the served model reads the engine
sources embedded in its own gguf plus the engine's BARO_PROFILE shares, and
proposes ONE unified diff per identity framing.

The output-format block is a placeholder skeleton, not a worked example:
iterations 002-005 saw 8 of 14 candidates echo the example back, invented
symbols and all (bench/loop-protocol.md, iteration 005 result).

Usage: tools/loop-propose.py MODEL.gguf ITER [--n 4] [--region auto|attn|ssm|ffn|head]
         [--profile .work/profile-run.log] [--endpoint http://127.0.0.1:8083] [--[no-]mega]
Writes .work/loop/<ITER>/{meta.json,FILES,src/,prompt.md,cand-<i>.raw.md,cand-<i>.diff,cand-<i>.predict}
"""
import argparse, json, re, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDENT = Path.home() / "Brain/identities/identities-index.md"
MARK = {"attn": ("# -- attention / ssm sub-block --", "# -- ffn sub-block --"),
        "ssm": ("# -- attention / ssm sub-block --", "# -- ffn sub-block --"),
        "ffn": ("# -- ffn sub-block --", "# -- head --"),
        "head": ("# -- head --", "# advance by the window")}
# Split layout under BARO_MEGA=1 (the default the gguf runs with): decode is one
# persistent megakernel, so the launch-path regions above are not on the executed
# path. The proposer is shown kernels/mega.mojo instead: the helper section (top
# of the file up to the first phase) plus the phase def for the target region,
# delimited by def boundaries -- kernel files carry no comment markers.
MEGA_DEF = {"attn": "attn_phases", "ssm": "ssm_phases", "ffn": "ffn_phases", "head": "mega_body"}
KFILES = {"attn": ["attn.mojo", "elementwise.mojo", "matmul_skinny.mojo"],
          "ssm": ["ssm.mojo", "elementwise.mojo", "matmul_skinny.mojo"],
          "ffn": ["elementwise.mojo", "matmul_skinny.mojo"],
          "head": ["elementwise.mojo", "matmul_skinny.mojo"]}

RULES = """You are optimising the inference engine embedded inside your own model file
(Qwythos-9B, AMD RX 7900 XTX gfx1100, Mojo). Decode is GPU-bound and
bandwidth-bound: single-token decode reads every weight once, so wins come from
fewer bytes moved, fewer kernel launches, and fused math -- not from FLOPs.
Propose exactly ONE change, one concern, as a unified diff against the files
shown (paths relative to the source dir, e.g. `--- a/matmul_skinny.mojo`). Rules:
- Only edit the files shown. Never touch anything else.
- Kernel files carry zero comments and zero docstrings.
- Output must stay bit-identical: the gate compares 64 greedy tokens.
- Do not change timing, printing, or profiling code.
- End with one line `PREDICT: <signed percent>` = your predicted change in tok/s_gen.
- Every symbol you reference must already exist in the sources shown. If your
  change needs a new kernel, you must write its full body in the same diff.
- Source files are given inside <file path="..."> tags. That is INPUT framing.
  Never use it in your answer -- your diff must use `--- a/<path>` / `+++ b/<path>`
  headers and @@ hunks, or it is discarded unparsed.

Output format, exactly:

<= 8 lines of rationale, then

```diff
--- a/<one of the files shown>
+++ b/<the same file>
@@ -<first line>,<count> +<first line>,<count> @@
 <an unchanged line, copied verbatim from the file>
-<the exact line you remove>
+<the line you put in its place>
 <an unchanged line, copied verbatim from the file>
```

then one line `PREDICT: <signed percent>`.

Every line inside the fence is a real line of the file shown or your real
replacement. No `...`, no angle-bracket placeholders left in, no name that is
absent from the binding table and the files shown."""


def identities():
    rows = []
    for line in IDENT.read_text().splitlines():
        m = re.match(r"\| (\d\d) \| \[\[([^\]]+)\]\] \| [^|]+ \| (.+?) \|\s*$", line)
        if m:
            rows.append((m.group(2), m.group(3).strip()))
    return rows


def bindings(src_dir):
    # The comptime aliases moved out of engine.mojo into registry.mojo (9b8a399);
    # engine.mojo's `# --- kernel bindings` block is now empty, so slicing it fed
    # the proposer nothing and iteration 001's whole candidate set invented
    # symbol names. Emit the real alias table instead, from wherever it lives.
    txt = ""
    for f in ("registry.mojo", "engine.mojo"):
        p = src_dir / f
        if not p.exists():
            continue
        rows = re.findall(r"^comptime (\w+) = (amar_\w+)", p.read_text(), re.M)
        if rows:
            txt += (f"\nkernel bindings defined in {f} -- these names, and ONLY these, "
                    f"are callable as ctx.enqueue_function[<name>]:\n"
                    + "\n".join(f"  {n} = {k}" for n, k in rows) + "\n")
    return txt


def slice_region(engine_src, region, src_dir, body_file="engine.mojo"):
    a, b = MARK[region]
    lines = engine_src.splitlines()
    ia = next(i for i, l in enumerate(lines) if a in l)
    ib = next(i for i, l in enumerate(lines) if b in l and i > ia)
    tbl = bindings(src_dir)
    if not tbl.strip():
        raise SystemExit("loop-propose: no kernel binding table found; refusing to prompt blind")
    return (tbl + f"\n{body_file} lines {ia+1}-{ib} (target region `{region}`):\n"
            + "\n".join(lines[ia:ib]))


def profile_shares(path):
    shares, mega = {}, {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"profile: (\w+) ([\d.]+) ([\d.]+)", line)
        if m:
            shares[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        # BARO_PROFILE=5, megakernel: one line, microseconds per phase family. The
        # same log also carries the launch-path `profile:` lines, which under
        # BARO_MEGA=1 attribute one kernel to whichever sync follows; the stamp
        # line wins when present.
        m = re.match(r"mega profile \(last token, us\): ssm sub-blocks ([\d.]+)\s+attn sub-blocks ([\d.]+)"
                     r"\s+ffn ([\d.]+)\s+head ([\d.]+)\s+total ([\d.]+)", line)
        if m:
            tot = float(m.group(5))
            for k, v in zip(("ssm", "attn", "ffn", "head"), m.groups()[:4]):
                mega[k] = (float(v) / 1e6, float(v) / tot)
    return mega or shares


def def_span(lines, name):
    d = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}["))
    i = d
    while i > 0 and lines[i - 1].startswith("@"):
        i -= 1
    j = next((k for k in range(d + 1, len(lines)) if re.match(r"^(@|def )", lines[k])), len(lines))
    while j > i and not lines[j - 1].strip():
        j -= 1
    return i, j


def slice_mega(mega_src, region):
    lines = mega_src.splitlines()
    h0, _ = def_span(lines, "ssm_phases")
    a, b = def_span(lines, MEGA_DEF[region])
    return (f"\nYou are editing device code inside the persistent decode megakernel "
            f"(kernels/mega.mojo): one workgroup per row block, `grid_barrier` between "
            f"phases, every weight read once per token. There are no kernel launches to "
            f"add or remove here; wins are fewer bytes, fewer barriers, fewer passes. "
            f"Only the helpers shown and the names already used in the excerpt are callable.\n"
            f"\nmega.mojo, target region `{region}` = `{MEGA_DEF[region]}` (lines {a+1}-{b}); "
            f"the excerpt below is lines 1-{h0} (helpers) and {a+1}-{b} of that file:\n",
            "\n".join(lines[:h0]) + "\n\n" + "\n".join(lines[a:b]))


def ask(endpoint, prompt, identity_line, max_tokens=4096):
    # RULES + corpus go in the system message and are IDENTICAL across branches,
    # so the ~25k-char prefix is prefilled once and reused (cache_prompt). The
    # identity varies only in the trailing user turn. Putting the framing first
    # diverges the prefix and forces a full re-prefill per branch (~26 min each
    # at 4.4 t/s) -- and buys nothing: each branch still sees only its own framing.
    body = {"messages": [{"role": "system", "content": RULES + "\n\n" + prompt},
                         {"role": "user", "content": "Framing for this attempt: " + identity_line
                          + "\n\nPropose your one change now, in the output format given."}],
            "temperature": 0.6, "max_tokens": max_tokens, "cache_prompt": True}
    req = urllib.request.Request(endpoint + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    ch = d["choices"][0]
    msg = ch["message"]
    # A reasoning model served with --reasoning-format deepseek puts its thinking
    # in reasoning_content and leaves content empty until it stops thinking. Read
    # content, fall back to the thinking, and hand the caller the finish reason
    # and token counts -- iteration 003's first run recorded 4/4 "no diff" when
    # what actually happened was that every branch burned its whole budget
    # thinking and the harness dropped the text.
    text = msg.get("content") or ""
    if not text.strip():
        text = msg.get("reasoning_content") or ""
    return text, ch.get("finish_reason"), d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("iter")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--region", default="auto")
    ap.add_argument("--profile", default=".work/profile-run.log")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8083")
    ap.add_argument("--start", type=int, default=0, help="first identity index")
    ap.add_argument("--mega", action=argparse.BooleanOptionalAction, default=None,
                    help="show the megakernel phase instead of the launch-path region (default: auto, on when the gguf carries mega.mojo + window.mojo)")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="answer budget per branch; a reasoning model needs room to think AND answer")
    a = ap.parse_args()
    out = ROOT / ".work/loop" / a.iter
    (out / "src").mkdir(parents=True, exist_ok=True)
    meta = json.loads(subprocess.run([str(ROOT / ".venv/bin/python3"), str(ROOT / "tools/gguf-extract.py"),
                                      a.model, "--meta"], capture_output=True, text=True, check=True).stdout)
    files = meta["baro.kernel.files"].split(",")
    (out / "FILES").write_text("\n".join(files) + "\n")
    (out / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k.startswith("baro.")}, indent=1))
    for f in files:
        dst = out / "src" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(meta[f"baro.kernel.src.{f}"])
    shares = profile_shares(a.profile)
    region = a.region if a.region != "auto" else max(shares, key=lambda k: shares[k][1])
    prof_txt = "\n".join(f"  {k}: {v[0]*1000:.1f} ms  ({v[1]*100:.1f}%)" for k, v in shares.items())
    # Split layout (2026-09-08): the per-window body lives in window.mojo and
    # engine.mojo (the stopwatch) is no longer embedded. Legacy ggufs still
    # carry the body inside engine.mojo.
    body_file = "window.mojo" if (out / "src/window.mojo").exists() else "engine.mojo"
    mega = a.mega if a.mega is not None else (out / "src/mega.mojo").exists() and body_file == "window.mojo"
    prompt = (f"Engine source commit in this gguf: {meta['baro.kernel.commit']}\n"
              f"GPU time per decode run, by sub-block ({'BARO_PROFILE=5, megakernel phases' if mega else 'BARO_PROFILE=1'}):\n{prof_txt}\n"
              f"Target region: `{region}` (largest share).\n\n")
    if mega:
        intro, excerpt = slice_mega((out / "src/mega.mojo").read_text(), region)
        prompt += intro + f'\n<file path="mega.mojo">\n' + excerpt + '\n</file>'
    else:
        prompt += slice_region((out / "src" / body_file).read_text(), region, out / "src", body_file)
        # Tag-delimited, deliberately NOT diff-shaped: the old `===== f =====` banner
        # taught iter-001 cand-2 to answer in banners instead of a unified diff, and
        # the gate discarded it unparsed (receipt: parse / "no diff fence").
        for f in KFILES[region]:
            prompt += f'\n\n<file path="{f}">\n' + (out / "src" / f).read_text() + f'\n</file>'
    (out / "prompt.md").write_text(prompt)
    ids = identities()[a.start:a.start + a.n]
    print(f"region={region} mega={mega} prompt_chars={len(prompt)} identities={[i[0] for i in ids]}", flush=True)
    for i, (name, line) in enumerate(ids):
        raw, finish, usage = ask(a.endpoint, prompt, line, a.max_tokens)
        (out / f"cand-{i}.raw.md").write_text(f"identity: {name}\n\n" + raw)
        body = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        m = re.search(r"```diff\n(.*?)```", body, flags=re.S)
        p = re.search(r"PREDICT:\s*([+-]?\d+(?:\.\d+)?)\s*%?", body)
        (out / f"cand-{i}.diff").write_text(m.group(1) if m else "")
        (out / f"cand-{i}.predict").write_text((p.group(1) if p else "none") + "\n")
        print(f"cand-{i} [{name}] diff={'yes' if m else 'NO'} lines={m.group(1).count(chr(10)) if m else 0} "
              f"predict={p.group(1) if p else 'none'} finish={finish} "
              f"completion_tok={usage.get('completion_tokens')}", flush=True)
        if finish == "length":
            print(f"  cand-{i}: budget exhausted at {a.max_tokens} tokens -- "
                  f"no diff here is a harness result, not a proposer result", flush=True)


if __name__ == "__main__":
    sys.exit(main())
