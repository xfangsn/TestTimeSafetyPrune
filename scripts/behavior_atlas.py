"""Behavior Atlas: localize several RLHF-instilled behaviors with BLADE
(Behavioral Localization via Activation-Difference Edges, a.k.a. the
signed-actdiff-edge score), then compare (a) how concentrated each is and
(b) how much their weight sets overlap.

Behaviors: refusal (existing edge.pt) + Anthropic model-written A/B evals
(sycophancy, corrigibility, power-seeking, survival-instinct, self-awareness,
wealth-seeking, myopic-reward). Weak behaviors (|pick-rate - 0.5| < MIN_BIAS)
are skipped automatically.

Outputs:
  data/weight_scores/atlas_<name>_edge.pt   (per-behavior edge scores)
  results/behavior_atlas.json
  results/behavior_atlas_concentration.png
  results/behavior_atlas_overlap.png
"""

import json
from pathlib import Path

import torch

from ttsafety.behaviors import (CATALOG, behavior_edge_scores, fetch_ab,
                                make_splits, pick_rate)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, random_scores_like,
                                   rank_weight_indices, selection_from_ranking)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCORES = DATA / "weight_scores"
LAYERS = list(range(7, 19))
COMPONENTS = "both"
SPARSITIES = [0.0001, 0.0005, 0.002, 0.005, 0.02]
MAX_FRACTION = 0.02
MIN_BIAS = 0.10          # keep behaviors where |pick-rate - 0.5| >= this
PPL_BUDGET = 0.05
NEAR_CHANCE = 0.55       # behavior-side pick-rate considered "removed"
K_MATCH = 207_618        # matched-k for the overlap matrix (refusal headline)


def compute_behavior(model, tokenizer, name, wiki, base_ppl, base_ppl_sweep):
    rows = fetch_ab(name, DATA / "behaviors")
    splits = make_splits(rows)
    rate_m, _ = pick_rate(model, tokenizer, splits["val"], "matching")
    side = "matching" if rate_m >= 0.5 else "not_matching"
    base_rate, base_margin = pick_rate(model, tokenizer, splits["val"], side)
    print(f"\n[{name}] n={len(rows)} val={len(splits['val'])} "
          f"baseline pick-rate(match)={rate_m:.3f} -> side={side} bias={base_rate:.3f}",
          flush=True)
    if abs(rate_m - 0.5) < MIN_BIAS:
        print(f"  skip (weak bias)", flush=True)
        return None

    score_path = SCORES / f"atlas_{name}_edge.pt"
    if score_path.exists():
        scores = torch.load(score_path, map_location="cpu", weights_only=False)["scores"]
        print("  (loaded cached scores)", flush=True)
    else:
        scores, _ = behavior_edge_scores(model, tokenizer, splits["train"], side,
                                         LAYERS, COMPONENTS)
        torch.save({"scores": scores, "side": side}, score_path)

    ranking = rank_weight_indices(scores, MAX_FRACTION)
    ranking_rnd = rank_weight_indices(random_scores_like(scores, 0), MAX_FRACTION)
    sweep = []
    for frac in SPARSITIES:
        sel = selection_from_ranking(ranking, frac)
        with pruned_weights(model, sel):
            rate, _ = pick_rate(model, tokenizer, splits["val"], side)
            ppl = teacher_forced_ppl(model, tokenizer, wiki, max_tokens=10_000)
        sel_r = selection_from_ranking(ranking_rnd, frac)
        with pruned_weights(model, sel_r):
            rate_r, _ = pick_rate(model, tokenizer, splits["val"], side)
        ppl_d = (ppl - base_ppl_sweep) / base_ppl_sweep
        sweep.append({"sparsity": frac, "n_pruned": sum(len(v) for v in sel.values()),
                      "pick_rate": rate, "random_pick_rate": rate_r, "ppl_delta": ppl_d})
        print(f"  s={frac:.4%} edge {rate:.3f} (rand {rate_r:.3f}) pplΔ {ppl_d:+.2%}",
              flush=True)

    # concentration = smallest sparsity reaching near-chance within ppl budget
    reached = [r for r in sweep if r["pick_rate"] <= NEAR_CHANCE
               and r["ppl_delta"] <= PPL_BUDGET]
    conc = min(reached, key=lambda r: r["sparsity"])["sparsity"] if reached else None
    return {"n_items": len(rows), "baseline_pick_match": rate_m, "side": side,
            "baseline_bias": base_rate, "baseline_margin": base_margin,
            "sweep": sweep, "concentration_sparsity": conc}


def overlap_matrix(names):
    """Matched top-k overlap (Jaccard + enrichment) across all behaviors."""
    # load scores, build global offsets from the first, compute top-k global idx
    def load(name):
        path = (SCORES / "edge.pt") if name == "refusal" else (SCORES / f"atlas_{name}_edge.pt")
        return torch.load(path, map_location="cpu", weights_only=False)["scores"]

    ref = load(names[0])
    offsets, cur = {}, 0
    for n in sorted(ref):
        offsets[n] = cur
        cur += ref[n].numel()
    pool = cur

    def topk_global(scores):
        vals, gidx = [], []
        for nm in sorted(scores):
            flat = scores[nm].float().flatten()
            cap = max(1, int(0.10 * flat.numel()))
            v, loc = torch.topk(flat, cap, largest=True, sorted=False)
            vals.append(v)
            gidx.append(loc.long() + offsets[nm])
        vals = torch.cat(vals); gidx = torch.cat(gidx)
        order = torch.topk(vals, K_MATCH, largest=True, sorted=False).indices
        return set(gidx[order].tolist())

    sets = {n: topk_global(load(n)) for n in names}
    expected = K_MATCH * K_MATCH / pool
    jac = {}; enr = {}
    for a in names:
        for b in names:
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            jac[f"{a}|{b}"] = inter / union if union else 0.0
            enr[f"{a}|{b}"] = (inter / expected) if a != b else float("nan")
    return {"pool": pool, "k": K_MATCH, "expected_if_independent": expected,
            "jaccard": jac, "enrichment": enr, "names": names}


def main():
    model, tokenizer = load_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    SCORES.mkdir(parents=True, exist_ok=True)

    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tokenizer, wiki)
    base_ppl_sweep = teacher_forced_ppl(model, tokenizer, wiki, max_tokens=10_000)
    print(f"baseline wikitext ppl {base_ppl:.3f} (10k-window {base_ppl_sweep:.3f})",
          flush=True)

    report = {"env": env_info(), "layers": LAYERS, "base_ppl": base_ppl,
              "base_ppl_sweep": base_ppl_sweep, "behaviors": {}}
    for name in CATALOG:
        res = compute_behavior(model, tokenizer, name, wiki, base_ppl, base_ppl_sweep)
        if res:
            report["behaviors"][name] = res

    del model
    torch.cuda.empty_cache()

    # overlap matrix over refusal + all kept behaviors
    names = ["refusal"] + list(report["behaviors"])
    report["overlap"] = overlap_matrix(names)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "behavior_atlas.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("\nsaved results/behavior_atlas.json", flush=True)
    make_plots(report)


def make_plots(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"(skip plots: {e})", flush=True)
        return

    # concentration curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, r in report["behaviors"].items():
        x = [s["sparsity"] * 100 for s in r["sweep"]]
        y = [s["pick_rate"] for s in r["sweep"]]
        ax.plot(x, y, "o-", label=f"{name} (base {r['baseline_bias']:.2f})")
    ax.axhline(0.5, ls=":", c="grey", label="chance")
    ax.set_xscale("log"); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("pruned fraction of target pool (%)")
    ax.set_ylabel("behavior-side pick-rate (val)")
    ax.set_title("Behavior atlas: concentration (how much pruning removes each behavior)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "behavior_atlas_concentration.png", dpi=130)
    print("saved results/behavior_atlas_concentration.png", flush=True)

    # overlap heatmap (enrichment vs chance)
    names = report["overlap"]["names"]
    enr = report["overlap"]["enrichment"]
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            v = enr[f"{a}|{b}"]
            M[i, j] = np.nan if a == b else v
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="magma")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}×", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmax(M) * 0.6 else "black",
                        fontsize=8)
    ax.set_title(f"Weight-set overlap enrichment vs chance (matched top-{K_MATCH:,})")
    fig.colorbar(im, label="× above chance")
    fig.tight_layout()
    fig.savefig(RESULTS / "behavior_atlas_overlap.png", dpi=130)
    print("saved results/behavior_atlas_overlap.png", flush=True)


if __name__ == "__main__":
    main()
