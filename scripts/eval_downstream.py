"""W1/W2: downstream-task evaluation of pruned models.

Usage:
  python scripts/eval_downstream.py --config base          # sanity gate
  python scripts/eval_downstream.py --config edge_s0.0005
  python scripts/eval_downstream.py --all                  # all 11 configs

Per-config partial results land in results/downstream/{config}.json; existing
files are skipped (resume-safe). Masks are rebuilt deterministically from the
cached scores/rankings in data/weight_scores/ using ttsafety.weight_prune;
for every pruned config the selected-entries-are-zero check is run inside the
pruning context before any evaluation.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from ttsafety.downstream import TASKS, evaluate_task  # noqa: E402
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl  # noqa: E402
from ttsafety.models import env_info, load_model  # noqa: E402
from ttsafety.weight_prune import (  # noqa: E402
    matrixwise_set_difference,
    pruned_weights,
    selection_from_ranking,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCORES = DATA / "weight_scores"
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "downstream"
PPL_TOKENS = 10_000

CONFIGS = [
    "base",
    "edge_s0.0001", "edge_s0.0005", "edge_s0.001",
    "wei_p0.0001_q0.0001", "wei_p0.0005_q0.0005", "wei_p0.001_q0.001",
    "wei_p0.01_q0.01",
    "random0_s0.0001", "random0_s0.0005", "random0_s0.001",
    "ratio_s0.0001",  # optional reference
    # Wei et al. 2026 signed SNIP set-difference (matched to edge tiers)
    "signed_p0.0001_q0.0001", "signed_p0.0005_q0.0005", "signed_p0.001_q0.001",
]

# plausibility gate for the unpruned base model (3B instruct, zero-shot)
BASE_MIN = {"arc_easy": ("acc_norm", 0.60), "arc_challenge": ("acc_norm", 0.30),
            "hellaswag": ("acc_norm", 0.50), "piqa": ("acc_norm", 0.70),
            "winogrande": ("acc", 0.55), "boolq": ("acc", 0.55)}


def load_score_dict(name):
    payload = torch.load(SCORES / f"{name}.pt", map_location="cpu",
                         weights_only=False)
    return payload["scores"]


def build_selection(config):
    """Rebuild the pruning selection for a config name (None for base)."""
    if config == "base":
        return None
    if config.startswith("wei_") or config.startswith("signed_"):
        prefix = "wei_" if config.startswith("wei_") else "signed_"
        parts = config.removeprefix(prefix).removeprefix("p").split("_q")
        p, q = float(parts[0]), float(parts[1])
        utility = load_score_dict("wei_utility_snip")
        if prefix == "wei_":
            safety = load_score_dict("wei_safety_snip")
        else:
            signed = load_score_dict("wei_safety_signed_snip")
            # most-negative signed scores facilitate the refusal response
            safety = {name: (-s.float()).to(torch.float16)
                      for name, s in signed.items()}
        return matrixwise_set_difference(
            safety, utility, safety_fraction=q, utility_fraction=p)
    rule, frac = config.rsplit("_s", 1)
    ranking = torch.load(SCORES / f"ranking_{rule}.pt", map_location="cpu",
                         weights_only=True)
    return selection_from_ranking(ranking, float(frac))


def verify_zeroed(model, selection):
    """Selected weight entries must be exactly zero inside the context."""
    modules = dict(model.named_modules())
    checked = 0
    with pruned_weights(model, selection):
        for name, idx in selection.items():
            module = modules.get(name)
            if module is None:
                matches = [m for k, m in modules.items() if k.endswith(name)]
                if len(matches) != 1:
                    raise KeyError(f"cannot resolve {name!r}")
                module = matches[0]
            flat = module.weight.view(-1)
            vals = flat[idx.to(flat.device, torch.long)]
            if not bool((vals == 0).all()):
                raise RuntimeError(f"{name}: selected entries not zeroed")
            checked += int(idx.numel())
    return checked


def run_config(model, tokenizer, wiki, base_ppl_10k, config):
    out_path = OUT_DIR / f"{config}.json"
    if out_path.exists():
        print(f"{config}: already done, skipping")
        return
    selection = build_selection(config)
    n_pruned = 0 if selection is None else sum(
        int(v.numel()) for v in selection.values())
    result = {"config": config, "n_pruned": n_pruned,
              "env": env_info(), "tasks": {}}
    if selection is not None:
        checked = verify_zeroed(model, selection)
        print(f"{config}: zero-check passed on {checked} selected weights",
              flush=True)
        result["zero_check_n"] = checked
        ctx = pruned_weights(model, selection)
    else:
        ctx = nullcontext()
    with ctx:
        for task in TASKS:
            r = evaluate_task(model, tokenizer, task)
            result["tasks"][task] = r
            print(f"  {task}: acc {r['acc']:.4f} acc_norm "
                  f"{r['acc_norm']:.4f} (n={r['n']})", flush=True)
        ppl = teacher_forced_ppl(model, tokenizer, wiki,
                                 max_tokens=PPL_TOKENS)
    result["wikitext_ppl_10k"] = ppl
    result["ppl_delta_pct_10k"] = ((ppl - base_ppl_10k) / base_ppl_10k * 100
                                   if config != "base" else 0.0)
    accs = [t["acc"] for t in result["tasks"].values()]
    norms = [t["acc_norm"] for t in result["tasks"].values()]
    result["mean_acc"] = sum(accs) / len(accs)
    result["mean_acc_norm"] = sum(norms) / len(norms)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out_path)
    print(f"{config}: mean acc {result['mean_acc']:.4f} acc_norm "
          f"{result['mean_acc_norm']:.4f} ppl {ppl:.2f} -> saved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    configs = CONFIGS if args.all else [args.config]
    if configs == [None]:
        ap.error("--config or --all required")

    model, tokenizer = load_model()
    wiki = load_wikitext_text()
    base_path = OUT_DIR / "base.json"
    base_ppl_10k = (json.loads(base_path.read_text())["wikitext_ppl_10k"]
                    if base_path.exists() else None)

    for config in configs:
        run_config(model, tokenizer, wiki, base_ppl_10k or 1.0, config)
        if config == "base":
            base = json.loads(base_path.read_text())
            base_ppl_10k = base["wikitext_ppl_10k"]
            failed = {t: (m, v, base["tasks"][t][m])
                      for t, (m, v) in BASE_MIN.items()
                      if base["tasks"][t][m] < v}
            if failed:
                raise SystemExit(
                    f"SANITY GATE FAILED on base model: {failed}. "
                    "Debug prompt formats before evaluating pruned configs.")
            print("base sanity gate passed: "
                  + json.dumps({t: round(base["tasks"][t][m], 4)
                                for t, (m, _) in BASE_MIN.items()}))


if __name__ == "__main__":
    main()
