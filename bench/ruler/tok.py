"""Load the engine pack's tokenizer (tools/gguf-tokenizer.py) for RULER sizing.

Not llama.cpp's tokenizer, not nemo/hf/openai -- ours, per the lane brief:
context size is measured in OUR tokenizer's tokens.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(pack):
    spec = importlib.util.spec_from_file_location("gt", ROOT / "tools/gguf-tokenizer.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    meta = gt.load_meta(pack)
    tokenizer = gt.load_tokenizer(pack)

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return tokenizer, meta, count
