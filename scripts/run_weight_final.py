"""W4: one-shot held-out evaluation of the locked weight-orthogonalization edit."""

import hashlib
import json
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from ttsafety.judge import is_refusal, refusal_rate
from ttsafety.models import env_info, load_model
from ttsafety.weight_edit import (
    orthogonalized_weights,
    project_residual_writes,
    random_unit_direction,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SWEEP = RESULTS / "sweep_weight_ortho.json"
OUT = RESULTS / "weight_ortho_final.json"
LOCK = RESULTS / "weight_ortho_selection_lock.json"
SAMPLES = DATA / "samples_weight_ortho_final.jsonl"
DIRECTIONS = DATA / "directions" / "refusal_llama32_3b_instruct.pt"
MAX_NEW_TOKENS = 128


def prompts(name: str) -> list[str]:
    return [
        row["instruction"]
        for row in load_jsonl(DATA / name)
    ]


def selection_hash(selection: dict) -> str:
    canonical = json.dumps(
        selection, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def plot_results(sweep: dict, final: dict) -> None:
    cells = [
        cell for cell in sweep["main_cells"].values()
        if cell.get("status") == "complete"
    ]
    colors = {"mlp": "tab:blue", "attn": "tab:orange", "both": "tab:green"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for component in ("mlp", "attn", "both"):
        group = [x for x in cells if x["components"] == component]
        ax.scatter(
            [x["ppl_delta_pct"] for x in group],
            [x["harmful_refusal"] for x in group],
            label=component,
            alpha=0.75,
            color=colors[component],
        )
    metrics = final["metrics"]
    ax.scatter(
        [metrics["ppl_delta_pct"]],
        [metrics["test_refusal_edited"]],
        marker="*",
        s=220,
        color="black",
        label="locked final (test)",
    )
    ax.axvline(5.0, linestyle=":", color="red", linewidth=1)
    ax.set_xlabel("WikiText PPL delta (%)")
    ax.set_ylabel("harmful refusal rate")
    ax.set_title("Weight orthogonalization Pareto landscape")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "weight_ortho_pareto.png", dpi=160)
    plt.close(fig)

    scopes = ("source-only", "L8-L14", "L7-L18", "L0-L27")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    xs = list(range(len(scopes)))
    for offset, component in enumerate(("mlp", "attn", "both")):
        values = [
            min(
                x["harmful_refusal"] for x in cells
                if x["components"] == component and x["scope"] == scope
            )
            for scope in scopes
        ]
        ax.bar(
            [x + (offset - 1) * width for x in xs],
            values,
            width,
            label=component,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(scopes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("best val refusal over lambda")
    ax.set_title("Component and destination-scope comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "weight_ortho_components.png", dpi=160)
    plt.close(fig)


def main():
    if OUT.exists():
        raise SystemExit(
            "weight_ortho_final.json already exists; refusing to reuse held-out test"
        )
    sweep = json.loads(SWEEP.read_text())
    selection = sweep.get("selection")
    if sweep.get("selection_status") != "locked-for-final-test" or not selection:
        raise SystemExit("W3 selection is not locked for final test")
    digest = selection_hash(selection)
    lock = {
        "selection_sha256": digest,
        "selection": selection,
        "sweep_sha256_before_test": hashlib.sha256(SWEEP.read_bytes()).hexdigest(),
    }
    LOCK.write_text(json.dumps(lock, indent=2))

    model, tokenizer = load_model()
    directions = torch.load(DIRECTIONS, map_location="cpu", weights_only=True)
    harmful_test = prompts("harmful_test.jsonl")
    harmless = prompts("harmless.jsonl")
    wiki = load_wikitext_text()
    layers = selection["destination_layers"]
    component = selection["components"]
    strength = selection["strength"]
    direction = (
        directions if selection["direction_mode"] == "per-layer"
        else directions[selection["source_layer"]]
    )
    factory = partial(
        project_residual_writes,
        model,
        direction,
        layers,
        component,
        strength,
    )

    print("W4: generating locked baseline on harmful_test and harmless ...", flush=True)
    base_harmful = generate_texts(
        model, tokenizer, harmful_test, MAX_NEW_TOKENS
    )
    base_harmless = generate_texts(
        model, tokenizer, harmless, MAX_NEW_TOKENS
    )
    base_ppl = teacher_forced_ppl(model, tokenizer, wiki, max_tokens=50_000)

    print("W4: generating selected per-layer weight edit ...", flush=True)
    with factory():
        edited_harmful = generate_texts(
            model, tokenizer, harmful_test, MAX_NEW_TOKENS
        )
        edited_harmless = generate_texts(
            model, tokenizer, harmless, MAX_NEW_TOKENS
        )
        edited_ppl = teacher_forced_ppl(
            model, tokenizer, wiki, max_tokens=50_000
        )
    kl = prompt_kl(
        model,
        tokenizer,
        harmless[:128],
        edit_context=factory,
        max_length=128,
        batch_size=2,
    )

    print("W4: evaluating three per-layer random-direction controls ...", flush=True)
    random_controls = []
    for seed in range(3):
        random_directions = {
            layer: random_unit_direction(model.config.hidden_size, seed * 1000 + layer)
            for layer in layers
        }
        random_factory = partial(
            project_residual_writes,
            model,
            random_directions,
            layers,
            component,
            strength,
        )
        with random_factory():
            outputs = generate_texts(
                model, tokenizer, harmful_test, MAX_NEW_TOKENS
            )
        random_controls.append({
            "seed": seed,
            "test_refusal": refusal_rate(outputs),
        })
        print(
            f"  random seed {seed}: test refusal "
            f"{random_controls[-1]['test_refusal']:.3f}",
            flush=True,
        )

    print("W4: hook/materialized behavior check on 16 prompts ...", flush=True)
    with factory():
        hook_small = generate_texts(
            model, tokenizer, harmful_test[:16], MAX_NEW_TOKENS
        )
    with orthogonalized_weights(
        model, direction, layers, component, strength
    ):
        materialized_small = generate_texts(
            model, tokenizer, harmful_test[:16], MAX_NEW_TOKENS
        )
    materialized_check = {
        "n": 16,
        "agreement": completion_agreement(
            tokenizer, hook_small, materialized_small
        ),
        "hook_refusal": refusal_rate(hook_small),
        "materialized_refusal": refusal_rate(materialized_small),
    }

    metrics = {
        "test_refusal_baseline": refusal_rate(base_harmful),
        "test_refusal_edited": refusal_rate(edited_harmful),
        "test_compliance_baseline": 1 - refusal_rate(base_harmful),
        "test_compliance_edited": 1 - refusal_rate(edited_harmful),
        "harmless_refusal_baseline": refusal_rate(base_harmless),
        "harmless_refusal_edited": refusal_rate(edited_harmless),
        "wikitext_ppl_baseline": base_ppl,
        "wikitext_ppl_edited": edited_ppl,
        "ppl_delta_pct": (edited_ppl - base_ppl) / base_ppl * 100,
        "harmless_kl": kl,
        "harmless_agreement": completion_agreement(
            tokenizer, base_harmless, edited_harmless
        ),
        "harmless_quality": completion_quality(
            tokenizer, edited_harmless
        ),
        "random_controls": random_controls,
        "random_mean_test_refusal": sum(
            x["test_refusal"] for x in random_controls
        ) / len(random_controls),
        "materialized_check": materialized_check,
    }
    report = {
        "config": {
            "model": "llama32_3b_instruct",
            "selection": selection,
            "selection_sha256": digest,
            "n_harmful_test": len(harmful_test),
            "n_harmless": len(harmless),
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "env": env_info(),
        "metrics": metrics,
    }
    OUT.write_text(json.dumps(report, indent=2))

    changed = [
        i for i, (base, edited) in enumerate(
            zip(base_harmful, edited_harmful)
        )
        if is_refusal(base) and not is_refusal(edited)
    ][:10]
    still_refused = [
        i for i, text in enumerate(edited_harmful) if is_refusal(text)
    ][:5]
    rows = []
    for index in changed + still_refused:
        rows.append({
            "split": "harmful_test",
            "index": index,
            "category": "changed-to-nonrefusal" if index in changed else "still-refused",
            "instruction": harmful_test[index],
            "baseline": base_harmful[index],
            "edited": edited_harmful[index],
        })
    for index in range(5):
        rows.append({
            "split": "harmless",
            "index": index,
            "category": "harmless",
            "instruction": harmless[index],
            "baseline": base_harmless[index],
            "edited": edited_harmless[index],
        })
    with SAMPLES.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    plot_results(sweep, report)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
