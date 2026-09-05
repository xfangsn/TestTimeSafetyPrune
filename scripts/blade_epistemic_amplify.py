"""Scheme A step 6 — AMPLIFY the epistemic-uncertainty behavior the RIGHT way (plan §3 op 1:
SUPPRESSOR REMOVAL). raw-alphaW was gain-not-injection (known 0.00->0.00). Here we instead zero the
weights that write AGAINST the uncertainty direction, s- = [-(r.W.Δμ)]_+ (flip r, then the proven
BLADE-G removal machinery). Hypothesis: removing suppressors INJECTS uncertainty on KNOWN-answer
prompts (0 -> >0), matching what activation-steering (+k @L22: 0->0.60) could do but raw-alphaW could not.

Suppressors may live in different layers than the positive-writers, so we score over ALL layers and let
the ranking choose (and also report the removal-ELS L*=[23,16] for comparison). Controls: random
suppressor selection + shuffled-r, to show any injection is specific.

Env: BLADE_MODEL, BLADE_LSTAR="23,16". Usage: BLADE_MODEL=Qwen/Qwen3-8B .venv/bin/python scripts/blade_epistemic_amplify.py
"""
import json
import os
from pathlib import Path

import torch

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   random_scores_like, selection_from_ranking)
from blade_epistemic_els import (qwen_wrap, last_token_moments, unc_rate, split_by_entity,
                                 COMPONENTS, PPL_TOKENS)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
L_STAR = [int(x) for x in os.environ.get("BLADE_LSTAR", "23,16").split(",")]
SCREEN = 0.03


def _medpos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap
    GEN.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))

    data = json.loads((RESULTS / "epistemic_pairs.json").read_text())["rows"]
    train, evl = split_by_entity(data)
    unc_tr = [r["question"] for r in train if r["label"] == 1]
    cert_tr = [r["question"] for r in train if r["label"] == 0]
    unc_ev = [r["question"] for r in evl if r["label"] == 1]
    cert_ev = [r["question"] for r in evl if r["label"] == 0]

    c4 = load_c4_text(); wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_hk = unc_rate(model, tok, cert_ev); base_hu = unc_rate(model, tok, unc_ev)
    print(f"{MODEL_ID} | base hedge KNOWN {base_hk:.3f} (inject target) unans {base_hu:.3f} | ppl {base_ppl:.2f}",
          flush=True)

    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    dirs_neg = {l: -directions[l] for l in directions}       # suppressor direction
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    print("generic-importance Q (g1scalar) ...", flush=True)
    Q, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4,
                                         seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=65536)
    lam = _medpos(score_edges(model, dirs_neg, muUNC, muCERT, all_layers, COMPONENTS)) / _medpos(Q)

    def supp_scores(dirs_, layers):
        S = score_edges_g(model, dirs_, muUNC, muCERT, layers, COMPONENTS, Q=Q, lam=lam, abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}

    def sel_from(scores, rho):
        return selection_from_ranking(rank_weight_indices(scores, max(SCREEN, rho)), rho)

    report = {"model": MODEL_ID, "L_star": L_STAR, "lam": lam, "op": "suppressor_removal",
              "base_hedge_known": base_hk, "base_hedge_unanswerable": base_hu,
              "base_ppl_c4": base_ppl, "base_ppl_wiki": base_ppl_wiki, "env": env_info(), "conditions": []}

    def run(label, sel):
        n = sum(int(v.numel()) for v in sel.values())
        with pruned_weights(model, sel):
            hk = unc_rate(model, tok, cert_ev); hu = unc_rate(model, tok, unc_ev)
            pc = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
            pw = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        row = {"label": label, "n_edges": n, "hedge_known": hk, "hedge_unanswerable": hu,
               "ppl_delta_c4": (pc - base_ppl) / base_ppl, "ppl_delta_wiki": (pw - base_ppl_wiki) / base_ppl_wiki}
        report["conditions"].append(row)
        print(f"  {label:26s} n={n:7d} hedge KNOWN {base_hk:.2f}->{hk:.2f}  unans {hu:.2f}  "
              f"ΔpplC4 {row['ppl_delta_c4']:+.2%}", flush=True)
        return row

    # suppressor removal over ALL layers (ranking picks), rho sweep
    print("== SUPPRESSOR REMOVAL (all layers, ranking picks) ==", flush=True)
    sc_all = supp_scores(dirs_neg, all_layers)
    for rho in (0.002, 0.005, 0.02, 0.05):
        run(f"supp_all_r{rho}", sel_from(sc_all, rho))
    # at the removal-ELS layers for comparison
    print("== SUPPRESSOR REMOVAL (removal-ELS L*) ==", flush=True)
    sc_ls = supp_scores(dirs_neg, L_STAR)
    for rho in (0.005, 0.02):
        run(f"supp_Lstar_r{rho}", sel_from(sc_ls, rho))
    # controls at a representative rho (all-layer)
    print("== CONTROLS (all layers, rho=0.02) ==", flush=True)
    for seed in (7, 17):
        run(f"ctrl_random_r0.02_s{seed}", sel_from(random_scores_like(sc_all, seed), 0.02))
    g = torch.Generator().manual_seed(123)
    dneg_shuf = {l: dirs_neg[l][torch.randperm(dirs_neg[l].numel(), generator=g)] for l in dirs_neg}
    run("ctrl_shuffledr_r0.02", sel_from(supp_scores(dneg_shuf, all_layers), 0.02))

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"blade_epistemic_amplify_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/blade_epistemic_amplify_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
