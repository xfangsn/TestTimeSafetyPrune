"""BLADE built from its NATURAL input (Anthropic sycophancy A/B contrastive pairs,
answer-span moments), full ELS+beta pipeline, then OOD-tested on TriviaQA
answer-sycophancy. This is the fair-to-BLADE version (each method uses its own
natural behavior representation): does BLADE's natural construction generalize OOD?
"""
import json
from pathlib import Path

import torch

from ttsafety.behaviors import (bestfirst_layers, collect_span_input_moments,
                                extract_direction, fetch_ab, make_splits, pick_rate,
                                score_edges, solo_layer_pool)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)
from ood_sycophancy_eval import load_ood, eval_ood

DATA = Path("data"); RESULTS = Path("results")
import os
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"; PPL_TOKENS = 5000
BETA = float(os.environ.get("BLADE_BETA", "0.05"))
EPS = 0.005; SCREEN_FRAC = 0.005; TESTFRAC = 0.005
RHOS = [0.0005, 0.002, 0.005, 0.02]
SUFFIX = f"_beta{int(BETA*100)}" if os.environ.get("BLADE_BETA") else ""


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    wiki = load_wikitext_text()
    sp = make_splits(fetch_ab("sycophancy", DATA / "behaviors"))
    val, train = sp["val"], sp["train"]
    exs = load_ood()

    rate_m, _ = pick_rate(model, tok, val, "matching")
    side = "matching" if rate_m >= 0.5 else "not_matching"
    other = "not_matching" if side == "matching" else "matching"
    print("BLADE direction/moments from Anthropic A/B answer spans ...", flush=True)
    directions = extract_direction(model, tok, train, side, eot=EOT)
    mu_a = collect_span_input_moments(model, tok, train, side, all_layers, COMPONENTS, eot=EOT)
    mu_b = collect_span_input_moments(model, tok, train, other, all_layers, COMPONENTS, eot=EOT)

    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_ab, _ = pick_rate(model, tok, val, side)

    def ppl_now():
        return teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)

    def measure():
        return pick_rate(model, tok, val, side)[0], ppl_now()

    pool = solo_layer_pool(model, directions, mu_a, mu_b, all_layers, COMPONENTS,
                           ppl_now, base_ppl, screen_frac=SCREEN_FRAC, beta=BETA)
    L_star = bestfirst_layers(model, directions, mu_a, mu_b, pool, COMPONENTS,
                              measure, base_ab, base_ppl, beta=BETA, eps=EPS,
                              test_frac=TESTFRAC)
    print(f"pool={len(pool)} -> L* = {L_star}", flush=True)

    base_ood = eval_ood(model, tok, exs)
    print(f"BASE  in-dist(A/B) {base_ab:.3f}  OOD syco {base_ood['sycophancy']:.3f} "
          f"acc {base_ood['accuracy']:.3f}", flush=True)
    rows = [{"rho": 0.0, "n_edges": 0, "indist_pick": base_ab,
             "ood_sycophancy": base_ood["sycophancy"], "ood_accuracy": base_ood["accuracy"],
             "ppl_delta": 0.0}]
    if L_star:
        scores = score_edges(model, directions, mu_a, mu_b, L_star, COMPONENTS)
        rk = rank_weight_indices(scores, max(0.03, max(RHOS)))
        for rho in RHOS:
            sel = selection_from_ranking(rk, rho)
            n = sum(int(v.numel()) for v in sel.values())
            with pruned_weights(model, sel):
                ab, _ = pick_rate(model, tok, val, side)
                ood = eval_ood(model, tok, exs)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            dppl = (ppl - base_ppl) / base_ppl
            rows.append({"rho": rho, "n_edges": n, "indist_pick": ab,
                         "ood_sycophancy": ood["sycophancy"], "ood_accuracy": ood["accuracy"],
                         "ppl_delta": dppl})
            print(f"BLADE(A/B) rho={rho:<7g} in-dist {ab:.3f} OOD syco {ood['sycophancy']:.3f} "
                  f"acc {ood['accuracy']:.3f} Δppl {dppl:+.1%} ({n:,} edges)", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ood_sycophancy_blade_ab{SUFFIX}.json").write_text(json.dumps(
        {"model": MODEL_ID, "source": "anthropic-A/B (natural)", "L_star": L_star,
         "beta": BETA, "n_ood": len(exs), "base_ppl": base_ppl, "rows": rows,
         "env": env_info()}, indent=2))
    print("saved results/ood_sycophancy_blade_ab.json", flush=True)


if __name__ == "__main__":
    main()
