"""Audit and sample the selected W5 ratio pruning cell on validation data only."""

import json
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import completion_quality
from ttsafety.generate import generate_texts
from ttsafety.judge import is_refusal, refusal_rate
from ttsafety.models import load_model
from ttsafety.weight_edit import iter_residual_writers
from ttsafety.weight_prune import (
    make_pruning_factory,
    selection_from_ranking,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "validate_weight_prune.json"
SAMPLES = DATA / "samples_weight_prune_val.jsonl"


def main():
    model, tokenizer = load_model()
    ranking = torch.load(
        DATA / "weight_scores" / "ranking_ratio.pt",
        map_location="cpu",
        weights_only=True,
    )
    selection = selection_from_ranking(ranking, 0.0001)
    writers = dict(iter_residual_writers(model, range(7, 19), "both"))
    before = {
        name: module.weight.view(-1)[indices.to(module.weight.device)].detach().cpu().clone()
        for name, indices in selection.items()
        for module in [writers[name]]
    }
    factory = make_pruning_factory(model, selection)
    harmful = [
        row["instruction"]
        for row in load_jsonl(DATA / "harmful_val.jsonl")
    ]
    harmless = [
        row["instruction"]
        for row in load_jsonl(DATA / "harmless.jsonl")
    ]
    baseline = generate_texts(model, tokenizer, harmful)
    with factory():
        pruned = generate_texts(model, tokenizer, harmful)
        harmless_pruned = generate_texts(model, tokenizer, harmless[:64])
        zero_verified = all(
            torch.count_nonzero(
                writers[name].weight.view(-1)[indices.to(writers[name].weight.device)]
            ).item() == 0
            for name, indices in selection.items()
        )
    restore_verified = all(
        torch.equal(
            before[name],
            writers[name].weight.view(-1)[
                indices.to(writers[name].weight.device)
            ].detach().cpu(),
        )
        for name, indices in selection.items()
    )
    report = {
        "config": {
            "rule": "ratio",
            "fraction": 0.0001,
            "split": "harmful_val",
            "test_not_used": True,
            "n_pruned": sum(x.numel() for x in selection.values()),
        },
        "baseline_refusal": refusal_rate(baseline),
        "pruned_refusal": refusal_rate(pruned),
        "harmless_first64_refusal": refusal_rate(harmless_pruned),
        "harmless_first64_quality": completion_quality(
            tokenizer, harmless_pruned
        ),
        "selected_values_zero_verified": zero_verified,
        "selected_values_restore_bit_exact": restore_verified,
    }
    OUT.write_text(json.dumps(report, indent=2))

    changed = [
        index for index, (left, right) in enumerate(zip(baseline, pruned))
        if is_refusal(left) and not is_refusal(right)
    ][:10]
    still = [
        index for index, text in enumerate(pruned) if is_refusal(text)
    ][:5]
    rows = []
    for index in changed + still:
        rows.append({
            "split": "harmful_val",
            "index": index,
            "category": "changed-to-nonrefusal" if index in changed else "still-refused",
            "instruction": harmful[index],
            "baseline": baseline[index],
            "pruned": pruned[index],
        })
    for index in range(5):
        rows.append({
            "split": "harmless",
            "index": index,
            "category": "harmless",
            "instruction": harmless[index],
            "pruned": harmless_pruned[index],
        })
    with SAMPLES.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

