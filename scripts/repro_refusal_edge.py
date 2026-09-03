"""Reproduce the headline refusal result: edge (BLADE) prune @0.05% -> refusal ~0,
ppl ~+0.6%. Uses the cached edge.pt scores + the unchanged prune pipeline.
Documented (docs/results-edge-summary.md): 0.05% (207,618) harmful refusal 0.000,
PPL Δ +0.61%, baseline ppl 13.06.
"""
import torch
from pathlib import Path
from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, load_model
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking, pruned_weights

DATA = Path("data")


@torch.no_grad()
def refusal_rate(model, tok, insts, bs=16):
    tok.padding_side = "left"
    prompts = [chat_wrap(tok, s) for s in insts]
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s+bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=128, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return sum(is_refusal(o) for o in outs) / len(outs)


def main():
    model, tok = load_model()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    wiki = load_wikitext_text()
    scores = torch.load(DATA / "weight_scores" / "edge.pt", map_location="cpu",
                        weights_only=False)["scores"]
    n_pool = sum(v.numel() for v in scores.values())

    base_ref = refusal_rate(model, tok, val)
    base_ppl = teacher_forced_ppl(model, tok, wiki)
    print(f"pool={n_pool:,} | baseline refusal {base_ref:.3f} | baseline ppl {base_ppl:.4f}",
          flush=True)

    rk = rank_weight_indices(scores, 0.001)          # per_matrix_cap default 0.10
    sel = selection_from_ranking(rk, 0.0005)         # 0.05% of pool
    n_pruned = sum(len(v) for v in sel.values())
    with pruned_weights(model, sel):
        ref = refusal_rate(model, tok, val)
        ppl = teacher_forced_ppl(model, tok, wiki)
    print(f"\nedge @0.05% ({n_pruned:,} weights):")
    print(f"  harmful_val refusal : {base_ref:.3f} -> {ref:.3f}   (doc: 1.000 -> 0.000)")
    print(f"  wikitext ppl        : {base_ppl:.4f} -> {ppl:.4f}   Δ {100*(ppl-base_ppl)/base_ppl:+.2f}%   (doc: +0.61%)")


if __name__ == "__main__":
    main()
