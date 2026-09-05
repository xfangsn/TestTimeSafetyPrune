"""Scheme A step 5 — REMOVE controls for the epistemic-uncertainty BLADE edit (Qwen3-8B).
Reuses the ELS layers found by blade_epistemic_els.py (default L*=[23,16]) to skip the search, and asks:
is the hedging drop SPECIFIC to the BLADE-selected weights + the true direction, or would any equal-size
(or equal-DAMAGE) edit do it?

Conditions, all at the ELS layers, measuring hedge-rate on unanswerable (want the DROP to be BLADE-only)
and known + C4/Wiki ppl:
  base | BLADE-G remove | random x3 (same sparsity) | shuffled-r (break r<->W) | random at higher
  sparsity (damage-matched: does random reach BLADE's hedge-drop only by doing much more ppl damage?).

Env: BLADE_MODEL, BLADE_LSTAR="23,16", BLADE_RHO=0.005.
Usage: BLADE_MODEL=Qwen/Qwen3-8B .venv/bin/python scripts/blade_epistemic_controls.py
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

# reuse the exact machinery from the ELS script (same wrap, moments, split, eval)
from blade_epistemic_els import (qwen_wrap, last_token_moments, unc_rate, split_by_entity,
                                 COMPONENTS, PPL_TOKENS)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
L_STAR = [int(x) for x in os.environ.get("BLADE_LSTAR", "23,16").split(",")]
RHO = float(os.environ.get("BLADE_RHO", "0.005"))
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
    train, evl = split_by_entity(data)                       # same seed -> same split as ELS run
    unc_tr = [r["question"] for r in train if r["label"] == 1]
    cert_tr = [r["question"] for r in train if r["label"] == 0]
    unc_ev = [r["question"] for r in evl if r["label"] == 1]
    cert_ev = [r["question"] for r in evl if r["label"] == 0]

    c4 = load_c4_text(); wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_hu = unc_rate(model, tok, unc_ev)
    base_hk = unc_rate(model, tok, cert_ev)
    print(f"{MODEL_ID} L*={L_STAR} rho={RHO} | base hedge unans {base_hu:.3f} known {base_hk:.3f} "
          f"| ppl {base_ppl:.2f}", flush=True)

    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    print("generic-importance Q (g1scalar) ...", flush=True)
    Q, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4,
                                         seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=65536)
    lam = _medpos(score_edges(model, directions, muUNC, muCERT, all_layers, COMPONENTS)) / _medpos(Q)

    def blade_scores(dirs_):
        S = score_edges_g(model, dirs_, muUNC, muCERT, L_STAR, COMPONENTS, Q=Q, lam=lam, abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}

    def sel_from(scores, rho):
        return selection_from_ranking(rank_weight_indices(scores, max(SCREEN, rho)), rho)

    blade = blade_scores(directions)
    shuf_g = torch.Generator().manual_seed(123)
    dirs_shuf = {l: directions[l][torch.randperm(directions[l].numel(), generator=shuf_g)]
                 for l in directions}
    blade_shuf = blade_scores(dirs_shuf)

    report = {"model": MODEL_ID, "L_star": L_STAR, "rho": RHO, "lam": lam,
              "base_hedge_unanswerable": base_hu, "base_hedge_known": base_hk,
              "base_ppl_c4": base_ppl, "base_ppl_wiki": base_ppl_wiki, "env": env_info(), "conditions": []}

    def run(label, sel):
        n = sum(int(v.numel()) for v in sel.values())
        with pruned_weights(model, sel):
            hu = unc_rate(model, tok, unc_ev); hk = unc_rate(model, tok, cert_ev)
            pc = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
            pw = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        row = {"label": label, "n_edges": n, "hedge_unanswerable": hu, "hedge_known": hk,
               "ppl_delta_c4": (pc - base_ppl) / base_ppl, "ppl_delta_wiki": (pw - base_ppl_wiki) / base_ppl_wiki}
        report["conditions"].append(row)
        print(f"  {label:22s} n={n:6d} hedge unans {base_hu:.2f}->{hu:.2f} known {hk:.2f} "
              f"ΔpplC4 {row['ppl_delta_c4']:+.2%}", flush=True)
        return row

    print("== BLADE + controls (all @ L*, rho unless noted) ==", flush=True)
    run("BLADE", sel_from(blade, RHO))
    for seed in (7, 17, 27):
        run(f"random_s{seed}", sel_from(random_scores_like(blade, seed), RHO))
    run("shuffled_r", sel_from(blade_shuf, RHO))
    # damage-matched: random at higher sparsity — does it reach BLADE's hedge-drop only via more ppl damage?
    for rho2 in (0.02, 0.05, 0.10):
        run(f"random_dmg_r{rho2}", sel_from(random_scores_like(blade, 7), rho2))

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"blade_epistemic_controls_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/blade_epistemic_controls_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
