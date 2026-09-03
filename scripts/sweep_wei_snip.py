"""Resume-safe Wei et al. SNIP set-difference pruning sweep."""

import argparse
import json
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import (
    completion_agreement,
    completion_quality,
    load_wikitext_text,
    prompt_kl,
    teacher_forced_ppl,
)
from ttsafety.generate import generate_texts
from ttsafety.judge import refusal_rate
from ttsafety.models import env_info, load_model
from ttsafety.weight_edit import iter_residual_writers
from ttsafety.weight_prune import make_pruning_factory, matrixwise_set_difference

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCORES = DATA / "weight_scores"
OUT = RESULTS / "sweep_wei_snip_set_difference.json"
BASE = RESULTS / "weight_edit_baseline.json"
BASE_COMPLETIONS = DATA / "cache" / "weight_edit_harmless_base.json"
MATCHED = tuple((x, x) for x in (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05))
PAPER_REGION = ((0.02, 0.01), (0.04, 0.02), (0.07, 0.03))
MAX_NEW_TOKENS = 128


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def load_scores(split: str):
    payload = torch.load(
        SCORES / f"wei_{split}_snip.pt",
        map_location="cpu",
        weights_only=False,
    )
    return payload["scores"], payload["metadata"]


def config_key(p: float, q: float) -> str:
    return f"wei_p{p:g}_q{q:g}"


def load_report(safety_meta, utility_meta):
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {
        "config": {
            "method": "Wei et al. (2024) wandg_set_difference",
            "official_code_commit": "0b0e7075d13f77621678976faa0fc66a4b2dffb4",
            "selection": "per-matrix flattened top-q safety minus top-p utility",
            "model": "llama32_3b_instruct",
            "layers": list(range(7, 19)),
            "components": "both",
            "matched_grid": MATCHED,
            "paper_region_grid": PAPER_REGION,
            "max_new_tokens": MAX_NEW_TOKENS,
            "controlled_adaptation": (
                "same residual-writer pool and evaluation as Signed edge/Taylor/Wanda"
            ),
        },
        "score_metadata": {"safety": safety_meta, "utility": utility_meta},
        "env": env_info(),
        "baseline": json.loads(BASE.read_text()),
        "cells": {},
    }


def evaluate(
    model, tokenizer, selection, harmful_val, harmless, base_outputs, wiki, base_ppl
):
    factory = make_pruning_factory(model, selection)
    with factory():
        harmful_outputs = generate_texts(
            model, tokenizer, harmful_val, MAX_NEW_TOKENS
        )
        harmless_outputs = generate_texts(
            model, tokenizer, harmless, MAX_NEW_TOKENS
        )
        ppl = teacher_forced_ppl(model, tokenizer, wiki, max_tokens=50_000)
    kl = prompt_kl(
        model,
        tokenizer,
        harmless[:128],
        edit_context=factory,
        max_length=128,
        batch_size=2,
    )
    return {
        "status": "complete",
        "harmful_refusal": refusal_rate(harmful_outputs),
        "harmless_refusal": refusal_rate(harmless_outputs),
        "wikitext_ppl": ppl,
        "ppl_delta_pct": (ppl - base_ppl) / base_ppl * 100,
        "harmless_kl": kl,
        "agreement": completion_agreement(tokenizer, base_outputs, harmless_outputs),
        "quality": completion_quality(tokenizer, harmless_outputs),
    }


def run(configs):
    safety, safety_meta = load_scores("safety")
    utility, utility_meta = load_scores("utility")
    if set(safety) != set(utility):
        raise ValueError("safety and utility score matrices do not match")
    report = load_report(safety_meta, utility_meta)
    model, tokenizer = load_model()
    harmful_val = [
        row["instruction"] for row in load_jsonl(DATA / "harmful_val.jsonl")
    ]
    harmless = [
        row["instruction"] for row in load_jsonl(DATA / "harmless.jsonl")
    ]
    base_outputs = json.loads(BASE_COMPLETIONS.read_text())
    wiki = load_wikitext_text()
    base_ppl = report["baseline"]["wikitext_ppl"]
    total_pool = sum(score.numel() for score in safety.values())

    for p, q in configs:
        key = config_key(p, q)
        if report["cells"].get(key, {}).get("status") == "complete":
            print(f"skip completed {key}", flush=True)
            continue
        print(f"building official matrix-global selection for {key} ...", flush=True)
        selection = matrixwise_set_difference(
            safety,
            utility,
            safety_fraction=q,
            utility_fraction=p,
        )
        per_matrix = {name: indices.numel() for name, indices in selection.items()}
        n_pruned = sum(per_matrix.values())
        safety_top_count = sum(
            max(1, int(score.numel() * q)) for score in safety.values()
        )
        utility_top_count = sum(
            max(1, int(score.numel() * p)) for score in utility.values()
        )
        print(
            f"  selected {n_pruned:,}/{total_pool:,} "
            f"({100 * n_pruned / total_pool:.5f}%)",
            flush=True,
        )
        try:
            cell = evaluate(
                model, tokenizer, selection, harmful_val, harmless,
                base_outputs, wiki, base_ppl,
            )
            cell.update({
                "key": key,
                "p_utility": p,
                "q_safety": q,
                "n_pruned": n_pruned,
                "actual_fraction": n_pruned / total_pool,
                "total_pool_weights": total_pool,
                "safety_top_count_before_difference": safety_top_count,
                "utility_top_count": utility_top_count,
                "protected_overlap_count": safety_top_count - n_pruned,
                "protected_overlap_fraction_of_safety_top": (
                    (safety_top_count - n_pruned) / safety_top_count
                ),
                "per_matrix_pruned": per_matrix,
                "max_matrix_fraction": max(
                    per_matrix[name] / module.weight.numel()
                    for name, module in iter_residual_writers(
                        model, range(7, 19), "both"
                    )
                ),
            })
        except Exception as exc:
            cell = {"key": key, "status": "failed", "error": repr(exc)}
            report["cells"][key] = cell
            atomic_json(OUT, report)
            raise
        report["cells"][key] = cell
        atomic_json(OUT, report)
        print(
            f"  refusal={cell['harmful_refusal']:.3f} "
            f"harmless={cell['harmless_refusal']:.3f} "
            f"ppl={cell['ppl_delta_pct']:+.2f}% KL={cell['harmless_kl']:.4f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid", choices=("matched", "paper", "all"), default="matched"
    )
    args = parser.parse_args()
    configs = {
        "matched": MATCHED,
        "paper": PAPER_REGION,
        "all": MATCHED + PAPER_REGION,
    }[args.grid]
    run(configs)


if __name__ == "__main__":
    main()
