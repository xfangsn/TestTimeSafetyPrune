"""TR-ELS decisive experiment (Phase B+C) on Llama-3.2-3B sycophancy, beta=10%.

Holds EVERYTHING fixed vs the plain-ELS beta=10% baseline (same data, same ppl
budget metric on WikiText, same test_frac) and changes ONLY the greedy objective,
to isolate whether transfer-robust selection avoids the L0 shortcut on merit.

Three variants of the layer-greedy objective:
  avg        : T = mean_e(gain_e)                  (reproduces plain best-first;
                                                    should pick the L0 shortcut)
  worst      : T = mean_e(a_e) - std_e(a_e)        (worst-slice / DRO, B)
  worst_ctrl : a_e = gain_target - max(gain_ctrl,0) then same T  (B + negative control, C)

Cross-fitting: for each of G environments, direction/moments are estimated from
the OTHER environments and the layer set is evaluated on the held-out environment.
Negative control = matched random mask (same layers, same k) via random_scores_like.
ppl budget check uses full-data direction on WikiText (identical to the baseline).
Then each variant's L* is OOD-tested on TriviaQA answer-sycophancy.
"""
import json
import os
import statistics as st
from pathlib import Path

import torch

from ttsafety.behaviors import (collect_span_input_moments, extract_direction,
                                fetch_ab, pick_rate, score_edges)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, random_scores_like,
                                   rank_weight_indices, selection_from_ranking)
from ood_sycophancy_eval import load_ood, eval_ood

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"; PPL_TOKENS = 5000
BETA = 0.10; TEST_FRAC = 0.005; EPS = 0.01; SEED = 0
G = 4; R = 2                       # environments; control masks per candidate
RHOS = [0.002, 0.005]
SEED_TAG = 12345


def make_envs(rows, g):
    """G heterogeneous slices by question-length quartile (a real distributional
    axis; both reviewers stressed slice heterogeneity is THE key variable)."""
    rows = sorted(rows, key=lambda r: len(r["question"]))
    n = len(rows); envs = []
    for i in range(g):
        envs.append(rows[i * n // g:(i + 1) * n // g])
    return envs


def worst_quartile(a):
    return st.mean(a) - (st.pstdev(a) if len(a) > 1 else 0.0)


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    wiki = load_wikitext_text()
    rows = fetch_ab("sycophancy", DATA / "behaviors")
    import random as _r
    _r.Random(SEED).shuffle(rows)
    rows = rows[:440]

    # global side
    side_probe = rows[:150]
    rate_m, _ = pick_rate(model, tok, side_probe, "matching")
    side = "matching" if rate_m >= 0.5 else "not_matching"
    other = "not_matching" if side == "matching" else "matching"
    print(f"side={side}", flush=True)

    envs = make_envs(rows, G)
    print("env sizes", [len(e) for e in envs], flush=True)

    # full-data direction/moments (for ppl-budget check + final edge scores + pool)
    full_dir = extract_direction(model, tok, rows, side, eot=EOT)
    full_a = collect_span_input_moments(model, tok, rows, side, all_layers, COMPONENTS, eot=EOT)
    full_b = collect_span_input_moments(model, tok, rows, other, all_layers, COMPONENTS, eot=EOT)

    # cross-fit: per env, estimate from the OTHER envs
    xdir, xa, xb = [], [], []
    for e in range(G):
        train_e = [r for j, env in enumerate(envs) if j != e for r in env]
        xdir.append(extract_direction(model, tok, train_e, side, eot=EOT))
        xa.append(collect_span_input_moments(model, tok, train_e, side, all_layers, COMPONENTS, eot=EOT))
        xb.append(collect_span_input_moments(model, tok, train_e, other, all_layers, COMPONENTS, eot=EOT))
        print(f"cross-fit env {e} estimated on {len(train_e)} rows", flush=True)

    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)

    def full_sel(cand):
        sc = score_edges(model, full_dir, full_a, full_b, cand, COMPONENTS)
        return selection_from_ranking(rank_weight_indices(sc, max(TEST_FRAC, 0.01)), TEST_FRAC)

    def ppl_ok(cand):
        with pruned_weights(model, full_sel(cand)):
            ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        return (ppl - base_ppl) / base_ppl <= BETA

    # --- candidate pool: solo prune within budget, per full-data direction (keeps L0 in play) ---
    pool = [l for l in all_layers if ppl_ok([l])]
    print(f"pool (solo Δppl<=β) = {pool}", flush=True)

    def env_pick(e, cand, ctrl_seed=None):
        sc = score_edges(model, xdir[e], xa[e], xb[e], cand, COMPONENTS)
        if ctrl_seed is None:
            sel = selection_from_ranking(rank_weight_indices(sc, max(TEST_FRAC, 0.01)), TEST_FRAC)
        else:
            rnd = random_scores_like(sc, seed=ctrl_seed)
            sel = selection_from_ranking(rank_weight_indices(rnd, max(TEST_FRAC, 0.01)), TEST_FRAC)
        with pruned_weights(model, sel):
            return pick_rate(model, tok, envs[e], side)[0]

    base_pi = [pick_rate(model, tok, envs[e], side)[0] for e in range(G)]
    print("base per-env pick", [round(x, 3) for x in base_pi], flush=True)

    def greedy(mode):
        S, cur = [], list(base_pi)
        while True:
            best_l, best_T = None, -1e9
            for l in pool:
                if l in S:
                    continue
                cand = sorted(S + [l])
                if not ppl_ok(cand):
                    continue
                a = []
                for e in range(G):
                    g_t = cur[e] - env_pick(e, cand)
                    if mode == "worst_ctrl":
                        gc = [cur[e] - env_pick(e, cand, ctrl_seed=SEED_TAG + 100 * e + q)
                              for q in range(R)]
                        a.append(g_t - max(st.median(gc), 0.0))
                    else:
                        a.append(g_t)
                T = st.mean(a) if mode == "avg" else worst_quartile(a)
                pos = sum(x > 0 for x in a) / G
                ok = (mode == "avg") or (pos >= 0.75)
                if ok and T > best_T:
                    best_l, best_T, best_a = l, T, a
            if best_l is not None and best_T > EPS:
                S.append(best_l)
                cur = [env_pick(e, sorted(S)) for e in range(G)]
                print(f"  [{mode}] +L{best_l} T={best_T:+.3f} per-env-pick={[round(x,2) for x in cur]}", flush=True)
            else:
                break
        return S

    results = {}
    exs = load_ood()
    base_ood = eval_ood(model, tok, exs)
    print(f"BASE OOD syco {base_ood['sycophancy']:.3f} acc {base_ood['accuracy']:.3f}", flush=True)

    for mode in ("avg", "worst", "worst_ctrl"):
        print(f"\n===== greedy mode={mode} =====", flush=True)
        L = greedy(mode)
        print(f"mode={mode}  L* = {L}", flush=True)
        rows_out = []
        if L:
            sc = score_edges(model, full_dir, full_a, full_b, L, COMPONENTS)
            rk = rank_weight_indices(sc, max(0.03, max(RHOS)))
            for rho in RHOS:
                sel = selection_from_ranking(rk, rho)
                n = sum(int(v.numel()) for v in sel.values())
                with pruned_weights(model, sel):
                    ood = eval_ood(model, tok, exs)
                    ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
                dppl = (ppl - base_ppl) / base_ppl
                rows_out.append({"rho": rho, "n_edges": n, "ood_sycophancy": ood["sycophancy"],
                                 "ood_accuracy": ood["accuracy"], "ppl_delta": dppl})
                print(f"  {mode} rho={rho} L*={L} OOD syco {ood['sycophancy']:.3f} "
                      f"acc {ood['accuracy']:.3f} Δppl {dppl:+.1%} ({n:,} edges)", flush=True)
        results[mode] = {"L_star": L, "rows": rows_out}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tr_els_sycophancy_beta10.json").write_text(json.dumps(
        {"model": MODEL_ID, "beta": BETA, "G": G, "R": R, "pool": pool,
         "base_ppl": base_ppl, "base_ood": base_ood, "base_per_env": base_pi,
         "results": results, "env": env_info()}, indent=2))
    print("\nsaved results/tr_els_sycophancy_beta10.json", flush=True)


if __name__ == "__main__":
    main()
