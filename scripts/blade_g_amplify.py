"""E2: does BLADE-G AMPLIFY strengthen OOD refusal at lower collateral (XSTest over-refusal, ppl) than
BLADE amplify? Amplify-optimal L*=[14]. Ranking S^amp ~ [c]_+ - lam*(alpha-1)*Q (from S^amp =
(alpha-1)[c]_+ - lam(alpha-1)^2 Q); amplify selected weights x alpha. Measure OOD refusal (HarmBench
prefill attack), XSTest over-refusal (full 250), WikiText ppl. Compare BLADE vs BLADE-G(g0,g1scalar).
Writes results/blade_g_amplify.json."""
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_nll
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments
from blade_refusal_amplify import scaled_weights
from blade_steering_baselines import (prefill_refusal, over_refusal, rel_ppl, wilson,
                                      PPL_TOKENS, C4_TOKENS)

DATA = Path("data"); RESULTS = Path("results")
L = [14]; COMP = "both"; N_DIR = 200; SCREEN = 0.03; Q_TOKENS = 65536
RHOS = [0.002, 0.005]; ALPHAS = [1.3, 1.5, 2.0, 2.5, 3.0]


def clean(S):
    return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}


def med_pos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]
    wiki = load_wikitext_text(); c4 = load_c4_text()
    allL = list(range(len(get_decoder_layers(model))))

    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:N_DIR], allL, COMP, chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], allL, COMP, chat_wrap)
    Qg0, _ = collect_c4_generic_importance(model, tok, L, COMP, text=c4, seqlen=2048, batch_size=2, mode="g0", max_tokens=Q_TOKENS)
    Qg1, meta = collect_c4_generic_importance(model, tok, L, COMP, text=c4, seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=Q_TOKENS)

    blade = score_edges(model, dirs, muH, muU, L, COMP)
    cmed = med_pos(blade); s1 = cmed / med_pos(Qg1); s0 = cmed / med_pos(Qg0)
    print(f"Q {meta} | scale_g1={s1:.2e} scale_g0={s0:.2e}", flush=True)

    base_nll_wiki, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_nll_c4, _ = teacher_forced_nll(model, tok, c4, max_tokens=C4_TOKENS)

    @contextmanager
    def noop(mode):
        yield
    base_ood = prefill_refusal(model, tok, ood, noop)
    base_benign = over_refusal(model, tok, xstest, noop)
    print(f"base OOD {base_ood:.3f} | XSTest over-refusal {base_benign:.3f}", flush=True)

    def amp_ctx(sel, a):
        @contextmanager
        def cm(mode):
            with scaled_weights(model, sel, a):
                yield
        return cm

    out = {"model": "meta-llama/Llama-3.2-3B-Instruct", "L": L, "rhos": RHOS, "Q_meta": meta,
           "base": {"ood_refusal": base_ood, "benign": base_benign}, "cells": [], "env": env_info()}
    for a in ALPHAS:
        # S^amp ranking uses lam_eff = lam*(alpha-1); BLADE uses plain c
        SCORES = {
            "BLADE": blade,
            "BLADE-G_g0": clean(score_edges_g(model, dirs, muH, muU, L, COMP, Q=Qg0, lam=s0 * (a - 1))),
            "BLADE-G_g1scalar": clean(score_edges_g(model, dirs, muH, muU, L, COMP, Q=Qg1, lam=s1 * (a - 1))),
        }
        ranks = {name: rank_weight_indices(S, SCREEN) for name, S in SCORES.items()}
        for rho in RHOS:
            for name, rk in ranks.items():
                sel = selection_from_ranking(rk, rho)
                ctx = amp_ctx(sel, a)
                n_ref = round(prefill_refusal(model, tok, ood, ctx) * len(ood)); oodr = n_ref / len(ood)
                benign = over_refusal(model, tok, xstest, ctx)
                relppl = rel_ppl(model, tok, wiki, base_nll_wiki, ctx, PPL_TOKENS)
                lo, hi = wilson(n_ref, len(ood))
                out["cells"].append({"score": name, "alpha": a, "rho": rho, "ood_refusal": oodr,
                                     "ci": [lo, hi], "benign": benign, "wiki_relppl": relppl})
                print(f"  a={a} rho={rho:<5} {name:18} OOD {oodr:.3f} XSTest {benign:.3f} "
                      f"Δppl {relppl:+.1%}", flush=True)
            RESULTS.mkdir(exist_ok=True)
            (RESULTS / "blade_g_amplify.json").write_text(json.dumps(out, indent=2))
    print("saved results/blade_g_amplify.json", flush=True)


if __name__ == "__main__":
    main()
