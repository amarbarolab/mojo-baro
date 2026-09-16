#!/usr/bin/env python3
"""Extract llama.cpp's pre-tokenizer table into serve/pretok-table.json.

usage: tools/pretok-table.py [LLAMACPP_SRC]        (default ~/Models/llama.cpp)

Sources, both read as text, no build: src/llama-vocab.cpp gives every tokenizer.ggml.pre
NAME its LLAMA_VOCAB_PRE_TYPE enum and per-name flags (clean_spaces, add_space_prefix,
ignore_merges, add_sep, escape_whitespaces, add_bos/add_eos when set there), and every
enum its regex_exprs list; convert_hf_to_gguf_update.py gives the HF repo and tokenizer
class the name was hashed from. Output is sorted and stable so the diff on a llama.cpp
bump is the change. tools/pretok-check.py compares serve/tokenizer.mojo against it.
"""
import json, pathlib, re, subprocess, sys
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "~/Models/llama.cpp").expanduser()
vocab = (src / "src/llama-vocab.cpp").read_text()
upd = (src / "convert_hf_to_gguf_update.py").read_text()
# 1. name -> enum + flags: the chain of `} else if (tokenizer_pre == "a" || ...) { pre_type = X; flag = v; }`
name2 = {}
for m in re.finditer(r"if\s*\(\s*((?:tokenizer_pre\s*==\s*\"[^\"]+\"\s*(?:\|\|\s*)?)+)\)\s*\{([^{}]*)\}", vocab):
    names = re.findall(r"tokenizer_pre\s*==\s*\"([^\"]+)\"", m.group(1)); body = m.group(2)
    enum = re.search(r"pre_type\s*=\s*(LLAMA_VOCAB_PRE_TYPE_\w+)", body)
    if not enum: continue
    flags = {k: (v == "true") for k, v in re.findall(r"\b(clean_spaces|add_space_prefix|ignore_merges|add_sep|escape_whitespaces|add_bos|add_eos)\s*=\s*(true|false)", body)}
    for n in names: name2[n] = {"enum": enum.group(1), "flags": flags}
# 2. enum -> regex list, from the switch: `case ENUM: (case ENUM:)* regex_exprs = { "..." , "..." };`
enum2 = {}
for m in re.finditer(r"((?:\s*case\s+LLAMA_VOCAB_PRE_TYPE_\w+:\s*)+)\s*regex_exprs\s*=\s*\{(.*?)\};", vocab, re.S):
    enums = re.findall(r"LLAMA_VOCAB_PRE_TYPE_\w+", m.group(1))
    body = re.sub(r"//[^\n]*", "", m.group(2))                       # drop commented originals
    regs = [json.loads('"' + s + '"') if "\\" in s else s for s in re.findall(r"\"((?:[^\"\\]|\\.)*)\"", body)]
    for e in enums: enum2[e] = regs
# 3. updater list: name -> repo, tokenizer type
upd2 = {}
for m in re.finditer(r"\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"tokt\"\s*:\s*TOKENIZER_TYPE\.(\w+)\s*,\s*\"repo\"\s*:\s*\"([^\"]+)\"", upd):
    upd2[m.group(1)] = {"tokt": m.group(2), "repo": m.group(3)}
m = re.search(r"\n\s*default:\s*(?://[^\n]*\s*)*regex_exprs\s*=\s*\{(.*?)\};", vocab, re.S)
if m:
    body = re.sub(r"//[^\n]*", "", m.group(1))
    enum2.setdefault("LLAMA_VOCAB_PRE_TYPE_DEFAULT", [json.loads('"' + x + '"') if "\\" in x else x for x in re.findall(r"\"((?:[^\"\\]|\\.)*)\"", body)])
table = {}
for n, d in sorted(name2.items()):
    table[n] = {"enum": d["enum"], "regexes": enum2.get(d["enum"], []), "flags": d["flags"], **upd2.get(n, {})}
commit = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out = {"source": "llama.cpp src/llama-vocab.cpp + convert_hf_to_gguf_update.py", "llama_cpp_commit": commit, "names": table}
pathlib.Path("serve/pretok-table.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
missing = [n for n, d in table.items() if not d["regexes"]]
print(f"pretok-table: {len(table)} names, {len(enum2)} regex sets, llama.cpp {commit}; names without a regex set: {missing}")
