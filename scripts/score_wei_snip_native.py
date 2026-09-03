"""Paper-native Wei et al. (2024) SNIP scoring over ALL decoder linear matrices (q/k/v/o_proj +
gate/up/down_proj, all 28 layers) -- the surface their released code prunes, vs. our controlled
o_proj+down_proj restriction. Same formula I(W,x)=mean_x|W * grad_W L(x)| (abs before average),
same 247 safety (caa_pairs refusal) / utility (harmless cached) examples as score_wei_snip.py.
Writes data/weight_scores/wei_{safety,utility}_snip_native.pt for a paper-native comparison panel."""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from ttsafety.models import env_info, load_model
from score_refusal_weights import response_batch
from score_wei_snip import safety_rows, utility_rows, atomic_torch_save

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "weight_scores"


def all_linears(model):
    """All decoder linear matrices, keyed like 'layers.<i>.<sub>' (model. prefix stripped)."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".layers." in name and mod.weight.ndim == 2:
            out[name[len("model."):] if name.startswith("model.") else name] = mod
    return out


def score_split(model, tokenizer, split, rows):
    writers = all_linears(model)
    model.requires_grad_(False)
    for m in writers.values():
        m.weight.requires_grad_(True)
    acc = {n: torch.zeros_like(m.weight, dtype=torch.float32) for n, m in writers.items()}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    for i, row in enumerate(rows, 1):
        model.zero_grad(set_to_none=True)
        nll = -response_batch(model, tokenizer, [row]).squeeze(0)
        nll.backward()
        with torch.no_grad():
            for n, m in writers.items():
                if m.weight.grad is None:
                    raise RuntimeError(f"missing grad for {n}")
                acc[n].add_((m.weight.detach().float() * m.weight.grad.detach().float()).abs_())
        if i % 20 == 0 or i == len(rows):
            print(f"Wei-native SNIP {split} {i}/{len(rows)} "
                  f"({(time.monotonic()-t0)/i:.2f}s/ex, peak "
                  f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB)", flush=True)
    scores = {n: (a / len(rows)).cpu() for n, a in acc.items()}
    meta = {"method": "wei2024_absolute_snip_native_all_linear",
            "split": split, "formula": "mean_x(abs(W*grad_W(mean_response_token_nll)))",
            "n_examples": len(rows), "n_matrices": len(scores),
            "target_pool": "ALL decoder linear matrices (q,k,v,o,gate,up,down); paper-native surface",
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(), "env": env_info()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"wei_{split}_snip_native.pt"
    atomic_torch_save({"scores": scores, "metadata": meta}, path)
    print(f"saved {path} ({len(scores)} matrices)", flush=True)
    model.zero_grad(set_to_none=True)
    del scores, acc
    torch.cuda.empty_cache()


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    safety, _ = safety_rows()
    utility, _ = utility_rows()
    print(f"safety {len(safety)} / utility {len(utility)} examples", flush=True)
    score_split(model, tok, "safety", safety)
    score_split(model, tok, "utility", utility)


if __name__ == "__main__":
    main()
