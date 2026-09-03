"""Does a high-perplexity BLADE removal actually hurt downstream tasks?
Measures 6 zero-shot tasks (acc_norm) at the deep, high-ppl removal points for
sycophancy (+19% ppl) and deception (+14% ppl) on Llama-3.2-3B, vs base.
If acc_norm is ~unchanged, WikiText ppl is a poor proxy and +20% ppl is not a
real downstream cost.
"""
import json
import os
from pathlib import Path

import torch

from ttsafety.behaviors import (collect_span_input_moments, extract_direction,
                                fetch_ab, make_splits, score_edges)
from ttsafety.downstream import TASKS, evaluate_task
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.models import env_info, load_model
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
EOT = "<|eot_id|>"; COMPONENTS = "both"; PPL_TOKENS = 10_000
# (tag, behavior, side, L*, rho) at low-beta (10%) vs high-beta (100%) points
CONFIGS = [
    ("sycophancy_b10", "sycophancy", "matching", [0, 2], 0.002),
    ("sycophancy_b100", "sycophancy", "matching", [0, 2, 1], 0.005),
    ("deception_b10", "deception", "not_matching", [10, 6, 2, 12, 21, 14, 0, 15, 16], 0.005),
    ("deception_b100", "deception", "not_matching", [1, 10, 4, 12, 21, 19, 8, 16, 9, 11], 0.02),
]


def build_selection(model, tok, name, side, layers, rho):
    rows = fetch_ab(name, DATA / "behaviors")
    sp = make_splits(rows)
    other = "not_matching" if side == "matching" else "matching"
    directions = extract_direction(model, tok, sp["train"], side, eot=EOT)
    mu_a = collect_span_input_moments(model, tok, sp["train"], side, layers, COMPONENTS, eot=EOT)
    mu_b = collect_span_input_moments(model, tok, sp["train"], other, layers, COMPONENTS, eot=EOT)
    scores = score_edges(model, directions, mu_a, mu_b, layers, COMPONENTS)
    rk = rank_weight_indices(scores, max(0.03, rho))
    return selection_from_ranking(rk, rho)


def run_tasks(model, tok):
    return {t: evaluate_task(model, tok, t) for t in TASKS}


def summarize(tag, tasks, ppl, base_ppl, base_tasks=None):
    norms = [t["acc_norm"] for t in tasks.values()]
    mean_norm = sum(norms) / len(norms)
    dppl = (ppl - base_ppl) / base_ppl * 100
    line = f"{tag:22} mean_acc_norm={mean_norm:.4f}  ppl={ppl:.2f} ({dppl:+.1f}%)"
    if base_tasks is not None:
        base_mean = sum(t["acc_norm"] for t in base_tasks.values()) / len(base_tasks)
        line += f"  Δacc_norm={(mean_norm - base_mean)*100:+.2f}pp"
    print(line, flush=True)
    for t in TASKS:
        d = ""
        if base_tasks is not None:
            d = f"  ({(tasks[t]['acc_norm']-base_tasks[t]['acc_norm'])*100:+.2f}pp)"
        print(f"    {t:14} acc_norm {tasks[t]['acc_norm']:.4f}{d}", flush=True)
    return {"mean_acc_norm": mean_norm, "ppl": ppl, "ppl_delta_pct": dppl, "tasks": tasks}


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    print("=== BASE ===", flush=True)
    base_tasks = run_tasks(model, tok)
    report = {"model": MODEL_ID, "env": env_info(),
              "base": summarize("base", base_tasks, base_ppl, base_ppl)}

    for tag, beh, side, layers, rho in CONFIGS:
        print(f"\n=== {tag}  L*={layers}  rho={rho:.1%} ===", flush=True)
        sel = build_selection(model, tok, beh, side, layers, rho)
        n = sum(int(v.numel()) for v in sel.values())
        print(f"  pruning {n:,} edges", flush=True)
        with pruned_weights(model, sel):
            tasks = run_tasks(model, tok)
            ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        report[tag] = summarize(tag, tasks, ppl, base_ppl, base_tasks)
        report[tag]["n_pruned"] = n

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_downstream_beta_compare.json").write_text(
        json.dumps(report, indent=2))
    print("\nsaved results/blade_downstream_beta_compare.json", flush=True)


if __name__ == "__main__":
    main()
