"""Measure ITI capability cost: C4 teacher-forced perplexity under the ITI steering hook (all positions),
for the same top-48 heads / directions used in baseline_dola_iti, at alpha in {2,6}. Saves results/iti_ppl.json.
Usage: BLADE_MODEL=Qwen/Qwen3-8B .venv/bin/python scripts/measure_iti_ppl.py"""
import json
import os
from pathlib import Path

import torch

from ttsafety.eval import load_c4_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model
from blade_epistemic_els import PPL_TOKENS
from baseline_dola_iti import head_acts, proj_std

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    blocks = get_decoder_layers(model)
    cfg = model.config
    nh, hd = cfg.num_attention_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    unc = [r["question"] for r in rows if r["label"] == 1]
    cert = [r["question"] for r in rows if r["label"] == 0]
    mu_u, _ = head_acts(model, tok, unc, blocks, nh, hd)
    mu_c, _ = head_acts(model, tok, cert, blocks, nh, hd)
    diffs = {(i, h): (mu_u[i][h] - mu_c[i][h]) for i in range(len(blocks)) for h in range(nh)}
    ranked = sorted(diffs, key=lambda k: -diffs[k].norm().item())[:48]
    dirs = {k: diffs[k] / diffs[k].norm().clamp_min(1e-6) for k in ranked}
    sigma = proj_std(model, tok, unc + cert, blocks, nh, hd, dirs)

    c4 = load_c4_text()
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    out = {"model": MODEL_ID, "base_ppl_c4": base_ppl, "alpha_ppl_delta": {}}

    def add_vec(alpha):
        add = {i: torch.zeros(nh * hd) for i in range(len(blocks))}
        for (i, h) in ranked:
            add[i].view(nh, hd)[h] = alpha * sigma[(i, h)] * dirs[(i, h)]
        return add

    for a in (2.0, 6.0):
        add = add_vec(a)
        handles = []

        def mk(i):
            vec = add[i]
            def hook(_m, args):
                return (args[0] + vec.to(args[0].dtype).to(args[0].device),) + tuple(args[1:])
            return hook
        for i in range(len(blocks)):
            handles.append(blocks[i].self_attn.o_proj.register_forward_pre_hook(mk(i)))
        try:
            p = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
        finally:
            for h in handles:
                h.remove()
        out["alpha_ppl_delta"][f"iti_a{a}"] = (p - base_ppl) / base_ppl
        print(f"ITI a{a}: ppl {p:.2f}  Δ {(p-base_ppl)/base_ppl:+.2%}", flush=True)

    (RESULTS / "iti_ppl.json").write_text(json.dumps(out, indent=2))
    print("saved results/iti_ppl.json")


if __name__ == "__main__":
    main()
