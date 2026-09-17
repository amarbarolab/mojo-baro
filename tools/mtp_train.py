#!/usr/bin/env python
"""A4 Order step 3: the 10-minute draft-head training smoke
(bench/draft-head-protocol.md). Fine-tunes MTPHead (tools/mtp_head.py) on
dense real-text next-token supervision from bench/draft_dump.mojo --mode
dump --env A4_TEXT_MODE=gsm8k (GSM8K train.jsonl question+answer text,
genuinely held out from both bench/data/e8_tasks.json and
bench/mtp-prompts/*.tokens).

One optimizer step PER DOCUMENT, not per shuffled position: blk.32's own
attention accumulates a KV cache across a document's positions (kc32_d /
vc32_d in serve/window.mojo), so a training batch that mixed shuffled
positions from different documents would feed the wrong causal context.
Each document's whole position sequence goes through MTPHead in one causal
forward (exactly serve/window.mojo's own multi-row verify-window shape),
loss is summed cross-entropy over all its positions, one backward per
document. Documents are capped so the total supervised-position count lands
near the smoke's frozen 2000, not per-document count.

Run only via gpu-wait:
  gpu-wait run --priority 10 --preemptible --vram 22 --timeout 900 -- \\
    PYTHONPATH=~/llama.cpp/gguf-py ~/AMDHQ/.venv/bin/python tools/mtp_train.py ...
"""
import argparse
import json
import struct
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "tools")
from mtp_head import MTPHead, build_causal_mask, load_trunk, H, read_dump  # noqa: E402


def flatten_document(doc, target_pairs_left):
    """From read_dump's per-document {tokens, h[n-1,H]}: row h[i] pairs
    with INPUT token tokens[i+1] and LABEL tokens[i+2], valid i in
    0..n-3 (bench/draft_dump.mojo's own docstring). Truncates to
    target_pairs_left if the document has more than that many positions
    left in the smoke's budget."""
    n = len(doc["tokens"])
    n_pairs = max(0, n - 2)  # h has n-1 rows, i ranges 0..n-3 -> n-2 pairs
    n_pairs = min(n_pairs, target_pairs_left)
    if n_pairs <= 0:
        return None
    h = doc["h"][:n_pairs]  # [n_pairs, H]
    input_ids = torch.tensor(doc["tokens"][1:1 + n_pairs], dtype=torch.long)
    labels = torch.tensor(doc["tokens"][2:2 + n_pairs], dtype=torch.long)
    return h, input_ids, labels


def train_one_document(model, head, opt, embed, lm_head, rotary, h, input_ids, labels, device):
    T = h.shape[0]
    h = h.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    tok_embed_row = embed(input_ids)
    pos_ids = torch.arange(1, 1 + T, device=device).view(1, 1, -1).expand(4, 1, -1)
    mask = build_causal_mask(T, torch.float32, device)
    opt.zero_grad(set_to_none=True)
    out = head(tok_embed_row, h, rotary, pos_ids, mask)
    logits = lm_head(out.to(lm_head.weight.dtype)).float()
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    opt.step()
    return loss.item(), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="bench/draft_dump.mojo --mode dump output")
    ap.add_argument("--extracted", required=True, help="tools/mtp_head.py --mode extract output")
    ap.add_argument("--hf-dir", default="$HOME/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf")
    ap.add_argument("--target-pairs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True, help="trained MTPHead state_dict checkpoint")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trunk(args.hf_dir)
    model.to(device)
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    rotary = model.model.rotary_emb if hasattr(model, "model") else model.language_model.rotary_emb
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config

    head = MTPHead(cfg).to(device)
    sd = torch.load(args.extracted, map_location=device)
    head.load_blk32(sd)
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    docs = read_dump(args.dump)
    torch.manual_seed(args.seed)
    perm = torch.randperm(len(docs)).tolist()

    pairs_left = args.target_pairs
    losses = []
    n_docs_used = 0
    n_pairs_used = 0
    peak_vram = 0
    for idx in perm:
        if pairs_left <= 0:
            break
        flat = flatten_document(docs[idx], pairs_left)
        if flat is None:
            continue
        h, input_ids, labels = flat
        loss, t = train_one_document(model, head, opt, embed, lm_head, rotary, h, input_ids, labels, device)
        losses.append(loss)
        n_docs_used += 1
        n_pairs_used += t
        pairs_left -= t
        if device == "cuda":
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())

    torch.save(head.state_dict(), args.out)
    report = {
        "n_docs_available": len(docs),
        "n_docs_used": n_docs_used,
        "n_pairs_used": n_pairs_used,
        "target_pairs": args.target_pairs,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "wall_s": time.time() - t_start,
        "peak_vram_gb": peak_vram / 2**30 if peak_vram else None,
        "checkpoint": args.out,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
