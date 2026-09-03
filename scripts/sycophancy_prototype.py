"""Prototype: transfer the signed-actdiff-edge localization from refusal to
sycophancy. Localize + prune "sycophancy weights"; check the behaviour drops
while capability (wikitext ppl) is preserved, with a random-weight control.

Pipeline (mirrors the refusal edge experiment):
  1. baseline sycophancy pick-rate (val/test)
  2. CAA sycophancy direction (train)
  3. sign-check the direction
  4. signed-actdiff-edge scores (r x W x (mu^S - mu^N))
  5. sparsity sweep on val: pick-rate + ppl, plus random-weight control
  6. report chosen cell on held-out test
Outputs: results/sycophancy_prototype.json + results/sycophancy_prototype.png
"""

import json
from pathlib import Path

import torch

from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.models import env_info, load_model
from ttsafety.sycophancy import (collect_input_moments, extract_sycophancy_direction,
                                 fetch_sycophancy, make_splits, score_edges,
                                 sycophancy_rate)
from ttsafety.weight_prune import (pruned_weights, random_scores_like,
                                   rank_weight_indices, selection_from_ranking)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LAYERS = list(range(7, 19))
COMPONENTS = "both"
SPARSITIES = [0.0001, 0.0005, 0.001, 0.005, 0.01]  # fraction of the target pool
PPL_BUDGET = 0.05        # <=5% wikitext ppl degradation
MAX_FRACTION = 0.02


def sign_check(model, tokenizer, directions, rows):
    """Projection of sycophantic answer spans should exceed honest spans."""
    from ttsafety.hooks import capture_span_mean
    from ttsafety.models import chat_wrap
    prompts = [chat_wrap(tokenizer, r["biased"]) for r in rows]
    syco = capture_span_mean(model, tokenizer, prompts, [r["matching"] for r in rows])
    hon = capture_span_mean(model, tokenizer, prompts, [r["not_matching"] for r in rows])
    out = {}
    for l in (10, 12, 14, 16):
        r = directions[l] / directions[l].norm()
        out[l] = {"syco_proj": float((syco[l] @ r).mean()),
                  "honest_proj": float((hon[l] @ r).mean())}
    return out


def main():
    model, tokenizer = load_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = fetch_sycophancy(DATA / "sycophancy")
    splits = make_splits(rows)
    print(f"data: {len(rows)} items -> "
          f"train {len(splits['train'])} val {len(splits['val'])} "
          f"test {len(splits['test'])}", flush=True)

    report = {"env": env_info(), "layers": LAYERS, "components": COMPONENTS,
              "n_train": len(splits["train"]), "n_val": len(splits["val"]),
              "n_test": len(splits["test"])}

    # 1. baseline
    base_val, m_val = sycophancy_rate(model, tokenizer, splits["val"])
    base_test, m_test = sycophancy_rate(model, tokenizer, splits["test"])
    print(f"baseline sycophancy pick-rate: val {base_val:.3f} (margin {m_val:+.2f}) "
          f"test {base_test:.3f}", flush=True)
    report["baseline"] = {"val_pick_rate": base_val, "val_margin": m_val,
                          "test_pick_rate": base_test, "test_margin": m_test}

    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tokenizer, wiki)
    report["baseline"]["wikitext_ppl"] = base_ppl
    print(f"baseline wikitext ppl: {base_ppl:.3f}", flush=True)

    # 2. direction
    print("extracting sycophancy direction ...", flush=True)
    directions = extract_sycophancy_direction(model, tokenizer, splits["train"])

    # 3. sign check
    report["sign_check"] = sign_check(model, tokenizer, directions, splits["val"])
    for l, d in report["sign_check"].items():
        print(f"  sign L{l}: syco {d['syco_proj']:+.2f} vs honest {d['honest_proj']:+.2f}",
              flush=True)

    # 4. edge scores
    print("collecting biased/neutral writer-input moments ...", flush=True)
    mu_s = collect_input_moments(model, tokenizer,
                                 [r["biased"] for r in splits["train"]],
                                 LAYERS, COMPONENTS)
    mu_n = collect_input_moments(model, tokenizer,
                                 [r["neutral"] for r in splits["train"]],
                                 LAYERS, COMPONENTS)
    scores = score_edges(model, directions, mu_s, mu_n, LAYERS, COMPONENTS)
    pool = sum(v.numel() for v in scores.values())
    report["target_pool_weights"] = pool
    print(f"edge scores over {pool:,} weights", flush=True)

    ranking = rank_weight_indices(scores, MAX_FRACTION)
    rnd = random_scores_like(scores, seed=0)
    ranking_rnd = rank_weight_indices(rnd, MAX_FRACTION)

    # 5. sweep
    print("sweeping sparsities (edge vs random) ...", flush=True)
    sweep = []
    for frac in SPARSITIES:
        sel = selection_from_ranking(ranking, frac)
        n_pruned = sum(len(v) for v in sel.values())
        with pruned_weights(model, sel):
            rate, margin = sycophancy_rate(model, tokenizer, splits["val"])
            ppl = teacher_forced_ppl(model, tokenizer, wiki, max_tokens=20_000)
        sel_r = selection_from_ranking(ranking_rnd, frac)
        with pruned_weights(model, sel_r):
            rate_r, _ = sycophancy_rate(model, tokenizer, splits["val"])
            ppl_r = teacher_forced_ppl(model, tokenizer, wiki, max_tokens=20_000)
        ppl_d = (ppl - base_ppl) / base_ppl
        row = {"sparsity": frac, "n_pruned": n_pruned,
               "val_pick_rate": rate, "val_margin": margin,
               "ppl_delta": ppl_d,
               "random_val_pick_rate": rate_r,
               "random_ppl_delta": (ppl_r - base_ppl) / base_ppl}
        sweep.append(row)
        print(f"  s={frac:.4%} n={n_pruned:>6} edge {rate:.3f} "
              f"(rand {rate_r:.3f}) pplΔ {ppl_d:+.2%}", flush=True)
    report["sweep"] = sweep

    # 6. choose: largest pick-rate drop within ppl budget, then test
    feasible = [r for r in sweep if r["ppl_delta"] <= PPL_BUDGET]
    chosen = min(feasible, key=lambda r: r["val_pick_rate"]) if feasible else None
    report["chosen"] = chosen
    if chosen:
        sel = selection_from_ranking(ranking, chosen["sparsity"])
        with pruned_weights(model, sel):
            t_rate, t_margin = sycophancy_rate(model, tokenizer, splits["test"])
            t_ppl = teacher_forced_ppl(model, tokenizer, wiki)
        report["test"] = {"sparsity": chosen["sparsity"],
                          "test_pick_rate": t_rate, "test_margin": t_margin,
                          "baseline_test_pick_rate": base_test,
                          "test_ppl_delta": (t_ppl - base_ppl) / base_ppl}
        print(f"\nCHOSEN s={chosen['sparsity']:.4%}: test sycophancy "
              f"{base_test:.3f} -> {t_rate:.3f}  (pplΔ "
              f"{(t_ppl - base_ppl) / base_ppl:+.2%})", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sycophancy_prototype.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("saved results/sycophancy_prototype.json", flush=True)
    make_plot(report)


def make_plot(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})", flush=True)
        return
    sw = report["sweep"]
    x = [r["sparsity"] * 100 for r in sw]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    base = report["baseline"]["test_pick_rate"]
    ax1.axhline(0.5, ls=":", c="grey", lw=1, label="chance (0.5)")
    ax1.axhline(report["baseline"]["val_pick_rate"], ls="--", c="#3d5a80",
                lw=1, label=f"baseline {report['baseline']['val_pick_rate']:.2f}")
    ax1.plot(x, [r["val_pick_rate"] for r in sw], "o-", c="#e07a5f",
             label="edge prune")
    ax1.plot(x, [r["random_val_pick_rate"] for r in sw], "s--", c="#81b29a",
             label="random prune")
    ax1.set_xscale("log")
    ax1.set_xlabel("pruned fraction of target pool (%)")
    ax1.set_ylabel("sycophancy pick-rate (val)")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=.3)
    ax2 = ax1.twinx()
    ax2.plot(x, [r["ppl_delta"] * 100 for r in sw], "^:", c="#8d6e63",
             label="ppl Δ% (edge)")
    ax2.set_ylabel("wikitext ppl Δ (%)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=8)
    ax1.set_title("Sycophancy edge-pruning: behaviour drop vs capability cost")
    fig.tight_layout()
    fig.savefig(RESULTS / "sycophancy_prototype.png", dpi=130)
    print("saved results/sycophancy_prototype.png", flush=True)


if __name__ == "__main__":
    main()
