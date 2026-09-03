"""Sweep edge-score variants (tasks #2/#3) at the three edge tiers.

Variants: edge_signcons, edge_trimmed, edge_subspace_k{2,4,8} (k=1 is the
baseline edge). Tiers: fraction in {0.0001, 0.0005, 0.001} of the 415M pool,
selection via the same capped global top-k as sweep_weight_prune.py
(per-matrix 10% candidate cap). Evaluation pipeline imported unchanged from
sweep_weight_prune (harmful_val 64 refusal, harmless 320, wikitext ppl 50k,
harmless KL, quality). Cells land in results/edge_variants.json after every
cell (resume-safe).
"""

import argparse
import json
from pathlib import Path

import torch

from sweep_weight_prune import evaluate  # noqa: E402  (script import: defs only)
from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCORES = DATA / "weight_scores"
OUT = RESULTS / "edge_variants.json"
BASE = RESULTS / "weight_edit_baseline.json"
BASE_COMPLETIONS = DATA / "cache" / "weight_edit_harmless_base.json"
VARIANTS = ("edge_signcons", "edge_trimmed",
            "edge_subspace_k2", "edge_subspace_k4", "edge_subspace_k8")
TIERS = (0.0001, 0.0005, 0.001)
MAX_FRACTION = max(TIERS)
SUBSPACE_KS = (2, 4, 8)
GAMMA = 1.0


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def get_ranking(variant: str):
    path = SCORES / f"ranking_{variant}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    print(f"building capped ranking for {variant} ...", flush=True)
    scores = torch.load(SCORES / f"{variant}.pt", map_location="cpu",
                        weights_only=False)["scores"]
    ranking = rank_weight_indices(scores, MAX_FRACTION, largest=True,
                                  per_matrix_cap=0.10)
    torch.save(ranking, path)
    return ranking


def load_report():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {
        "config": {
            "variants": list(VARIANTS),
            "tiers": list(TIERS),
            "baseline_rule": "edge (results/sweep_weight_prune.json)",
            "layers": list(range(7, 19)),
            "components": "both",
            "per_matrix_cap": 0.10,
            "max_new_tokens": 128,
        },
        "env": env_info(),
        "baseline": json.loads(BASE.read_text()),
        "cells": {},
    }


def run(variants):
    report = load_report()
    model, tokenizer = load_model()
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    base_outputs = json.loads(BASE_COMPLETIONS.read_text())
    wiki = load_wikitext_text()
    base_ppl = report["baseline"]["wikitext_ppl"]

    for variant in variants:
        ranking = get_ranking(variant)
        for fraction in TIERS:
            key = f"{variant}_s{fraction:g}"
            if report["cells"].get(key, {}).get("status") == "complete":
                print(f"skip completed {key}", flush=True)
                continue
            print(f"evaluating {key} ...", flush=True)
            selection = selection_from_ranking(ranking, fraction)
            try:
                cell = evaluate(
                    model, tokenizer, selection, harmful_val, harmless,
                    base_outputs, wiki, base_ppl,
                )
                cell.update({
                    "key": key,
                    "variant": variant,
                    "fraction": fraction,
                    "n_pruned": sum(int(v.numel())
                                    for v in selection.values()),
                    "total_pool_weights": ranking["total_pool_weights"],
                })
            except Exception as exc:
                cell = {"key": key, "status": "failed", "error": repr(exc)}
                report["cells"][key] = cell
                atomic_json(OUT, report)
                raise
            report["cells"][key] = cell
            atomic_json(OUT, report)
            print(f"  refusal={cell['harmful_refusal']:.3f} "
                  f"harmless={cell['harmless_refusal']:.3f} "
                  f"ppl={cell['ppl_delta_pct']:+.2f}% "
                  f"KL={cell['harmless_kl']:.4f}", flush=True)


def finalize():
    import numpy as np

    report = load_report()
    sweep = json.loads((RESULTS / "sweep_weight_prune.json").read_text())
    base_cells = sweep["cells"]
    # baseline edge + cached random0 rows for comparison
    report["baseline_cells"] = {
        k: {"harmful_refusal": c["harmful_refusal"],
            "harmless_refusal": c["harmless_refusal"],
            "ppl_delta_pct": c["ppl_delta_pct"],
            "harmless_kl": c["harmless_kl"],
            "n_pruned": c["n_pruned"]}
        for k, c in base_cells.items()
        if k.startswith("edge_s") or k.startswith("random0_s")
    }
    # diagnostics: PCA stats + sign-consistency stats
    pca = json.loads((SCORES / "pca_subspace_directions.json").read_text())
    report["diagnostics"] = {"pca": pca["stats"]}
    stats_path = SCORES / "edge_signcons_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())["per_matrix"]
        med = [s["consistency_quantiles_on_positive"][0] for s in stats]
        q99 = [s["consistency_quantiles_on_positive"][2] for s in stats]
        report["diagnostics"]["signcons"] = {
            "gamma": GAMMA,
            "median_consistency_median": float(np.median(med)),
            "median_consistency_q99": float(np.median(q99)),
            "note": "per-matrix quantiles of n_pos/256 over positive edges",
        }
    # clamp-then-sum diagnostic: subsampled correlation with sum-then-clamp
    clamp_diag = {}
    rng = torch.Generator().manual_seed(0)
    for k in SUBSPACE_KS:
        a = torch.load(SCORES / f"edge_subspace_k{k}.pt", map_location="cpu",
                       weights_only=False)["scores"]
        b = torch.load(SCORES / f"edge_subspace_k{k}_clampfirst.pt",
                       map_location="cpu", weights_only=False)["scores"]
        cors = []
        for name in sorted(a):
            fa, fb = a[name].float().flatten(), b[name].float().flatten()
            idx = torch.randperm(fa.numel(), generator=rng)[:2_000_000]
            cors.append(float(np.corrcoef(fa[idx].numpy(),
                                          fb[idx].numpy())[0, 1]))
        clamp_diag[f"k{k}"] = {"mean_pearson_vs_sum_then_clamp":
                               float(np.mean(cors))}
    report["diagnostics"]["clamp_first_vs_sum_then_clamp"] = clamp_diag

    # per-tier verdict vs baseline edge
    verdicts = []
    for fraction in TIERS:
        base = base_cells[f"edge_s{fraction:g}"]
        for variant in VARIANTS:
            cell = report["cells"][f"{variant}_s{fraction:g}"]
            better_refusal = cell["harmful_refusal"] < base["harmful_refusal"]
            not_worse_damage = (
                cell["ppl_delta_pct"] <= base["ppl_delta_pct"] + 1e-9
                and cell["harmless_kl"] <= base["harmless_kl"] + 1e-9)
            same_refusal_less_damage = (
                cell["harmful_refusal"] <= base["harmful_refusal"] + 1e-9
                and cell["ppl_delta_pct"] < base["ppl_delta_pct"] - 1e-9
                and cell["harmless_kl"] < base["harmless_kl"] - 1e-9)
            verdicts.append({
                "variant": variant, "fraction": fraction,
                "refusal": cell["harmful_refusal"],
                "baseline_refusal": base["harmful_refusal"],
                "ppl_delta_pct": cell["ppl_delta_pct"],
                "baseline_ppl_delta_pct": base["ppl_delta_pct"],
                "harmless_kl": cell["harmless_kl"],
                "baseline_harmless_kl": base["harmless_kl"],
                "beats_baseline": bool(
                    (better_refusal and not_worse_damage)
                    or same_refusal_less_damage),
            })
    report["verdicts_vs_edge"] = verdicts
    atomic_json(OUT, report)
    for v in verdicts:
        print(f"{v['variant']:22s} s{v['fraction']:g}: refusal "
              f"{v['refusal']:.3f} (edge {v['baseline_refusal']:.3f}) "
              f"ppl {v['ppl_delta_pct']:+.2f}% (edge "
              f"{v['baseline_ppl_delta_pct']:+.2f}%) KL {v['harmless_kl']:.3f} "
              f"(edge {v['baseline_harmless_kl']:.3f}) "
              f"{'BEATS' if v['beats_baseline'] else ''}")
    print("PC1 cos with refusal dir:",
          {s["layer"]: round(s["cos_pc1_refusal_direction"], 3)
           for s in pca["stats"]})
    print("clamp-first diagnostic:", clamp_diag)

    # figure: refusal and ppl vs n_pruned for all variants + edge + random0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    series = [("edge", None, "o", "black")] + [
        (v, v, m, None) for v, m in
        zip(VARIANTS, ("s", "D", "v", "P", "X"))] + [
        ("random0", None, "^", "gray")]
    random_cells = {k: c for k, c in base_cells.items()
                    if k.startswith("random0_s")}
    for label, variant, marker, color in series:
        if label == "edge":
            rows = [(base_cells[f"edge_s{f:g}"]["n_pruned"],
                     base_cells[f"edge_s{f:g}"]) for f in TIERS]
        elif label == "random0":
            rows = [(c["n_pruned"], c) for c in random_cells.values()]
        else:
            rows = [(report["cells"][f"{variant}_s{f:g}"]["n_pruned"],
                     report["cells"][f"{variant}_s{f:g}"]) for f in TIERS]
        rows.sort()
        axes[0].plot([r[0] for r in rows],
                     [r[1]["harmful_refusal"] for r in rows],
                     marker=marker, label=label, color=color)
        axes[1].plot([r[0] for r in rows],
                     [r[1]["ppl_delta_pct"] for r in rows],
                     marker=marker, label=label, color=color)
    for ax, title, ylabel in (
            (axes[0], "harmful_val refusal vs n_pruned", "refusal"),
            (axes[1], "wikitext ppl delta vs n_pruned", "ppl delta (%)")):
        ax.set_xscale("log")
        ax.set_xlabel("n_pruned (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Edge-score variants (tasks #2/#3) vs baseline edge")
    fig.tight_layout()
    fig.savefig(RESULTS / "edge_variants.png", dpi=150)
    print(f"saved {RESULTS / 'edge_variants.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
        return
    run(args.variants.split(","))


if __name__ == "__main__":
    main()
