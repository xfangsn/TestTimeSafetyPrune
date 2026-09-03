"""M4: strict CAA contrast-pair direction vs the mean-diff direction.

Stages (run in order, each persists its artifacts):
  --stage pairs      generate refusal/compliance responses on harmful_train,
                     filter, save data/caa_pairs.jsonl
  --stage extract    capture response-mean activations per pair, build
                     v_caa = mean(refusal - compliance), sign check, cosine
                     vs mean-diff -> data/directions/caa_*.pt + results JSON
  --stage spotcheck  steer with v_caa on harmful_val at the top-aligned
                     layers, compare with mean-diff sweep numbers
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ttsafety.data import load_jsonl
from ttsafety.generate import generate_texts
from ttsafety.hooks import capture_span_mean
from ttsafety.judge import is_refusal, refusal_rate
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.steer import steer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_TAG = "llama32_3b_instruct"
PAIRS_PATH = DATA_DIR / "caa_pairs.jsonl"
CAA_PT = DATA_DIR / "directions" / f"caa_{MODEL_TAG}.pt"
MD_PT = DATA_DIR / "directions" / f"refusal_{MODEL_TAG}.pt"
OUT_JSON = RESULTS_DIR / "caa_contrast.json"

GEN_LAYER, GEN_ALPHA = 14, -8.0  # strong jailbreak cell for compliance gens
SPOTCHECK_ALPHAS = [2.0, 4.0, 8.0]
MIN_WORDS = 20
MIN_UNIQUE_RATIO = 0.3


def _good_compliance(text: str) -> bool:
    if is_refusal(text):
        return False
    words = text.split()
    if len(words) < MIN_WORDS:
        return False
    return len(set(w.lower() for w in words)) / len(words) >= MIN_UNIQUE_RATIO


def stage_pairs(model, tokenizer, md_dirs, harmful_train, gen_alpha):
    print(f"Generating refusals (alpha=0) on {len(harmful_train)} harmful_train ...")
    refusals = generate_texts(model, tokenizer, harmful_train)
    print(f"Generating compliances (L{GEN_LAYER} alpha={gen_alpha}) ...")
    with steer(model, md_dirs[GEN_LAYER], layer=GEN_LAYER, alpha=gen_alpha):
        compliances = generate_texts(model, tokenizer, harmful_train)

    n_ref = sum(is_refusal(t) for t in refusals)
    n_comp = sum(_good_compliance(t) for t in compliances)
    print(f"confirmed refusals: {n_ref}/{len(harmful_train)}")
    print(f"confirmed quality compliances: {n_comp}/{len(harmful_train)}")
    print("sample compliances:")
    for t in compliances[:5]:
        print(f"  good={_good_compliance(t)}  {t[:160]!r}")

    pairs = [
        {"instruction": s, "refusal": r, "compliance": c}
        for s, r, c in zip(harmful_train, refusals, compliances)
        if is_refusal(r) and _good_compliance(c)
    ]
    with PAIRS_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"saved {len(pairs)} pairs to {PAIRS_PATH}")
    if len(pairs) < 100:
        print("WARNING: <100 pairs; consider relaxing the quality filter")


def stage_extract(model, tokenizer):
    pairs = load_jsonl(PAIRS_PATH)
    md_dirs = torch.load(MD_PT, weights_only=True)
    prompts = [chat_wrap(tokenizer, p["instruction"]) for p in pairs]
    print(f"Capturing response-mean activations for {len(pairs)} pairs ...")
    acts_ref = capture_span_mean(
        model, tokenizer, prompts, [p["refusal"] for p in pairs])
    acts_comp = capture_span_mean(
        model, tokenizer, prompts, [p["compliance"] for p in pairs])
    caa = {l: acts_ref[l].mean(0) - acts_comp[l].mean(0) for l in acts_ref}
    CAA_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(caa, CAA_PT)
    print(f"saved {len(caa)} CAA layer vectors to {CAA_PT}")

    # sign check: refusal response means must project higher (in-sample)
    # cosine similarity vs mean-diff direction
    report = {"n_pairs": len(pairs), "env": env_info(),
              "sign_check": {}, "cosine_vs_meandiff": {}}
    for layer in sorted(caa):
        v = caa[layer]
        v_hat = v / v.norm()
        p_ref = (acts_ref[layer] @ v_hat).mean().item()
        p_comp = (acts_comp[layer] @ v_hat).mean().item()
        md = md_dirs[layer]
        cos = torch.nn.functional.cosine_similarity(v, md, dim=0).item()
        report["sign_check"][str(layer)] = {
            "refusal_proj": p_ref, "compliance_proj": p_comp,
            "refusal_higher": bool(p_ref > p_comp),
        }
        report["cosine_vs_meandiff"][str(layer)] = cos
        print(f"  L{layer:2d}: cos(v_caa, v_md)={cos:+.3f} | "
              f"proj ref {p_ref:+.2f} vs comp {p_comp:+.2f}")

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"saved {OUT_JSON}")


def stage_spotcheck(model, tokenizer):
    report = json.loads(OUT_JSON.read_text())
    caa = torch.load(CAA_PT, weights_only=True)
    sweep = json.loads((RESULTS_DIR / "sweep_steer.json").read_text())
    harmful_val = [r["instruction"] for r in load_jsonl(DATA_DIR / "harmful_val.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA_DIR / "harmless.jsonl")]

    cos = report["cosine_vs_meandiff"]
    swept = set(sweep["config"]["layers"])
    top = sorted((l for l in map(int, cos) if l in swept),
                 key=lambda l: cos[str(l)], reverse=True)[:3]
    print(f"top-aligned swept layers: {top}")

    cells = {}
    for layer in top:
        for mag in SPOTCHECK_ALPHAS:
            with steer(model, caa[layer], layer=layer, alpha=-mag):
                r_harmful = refusal_rate(generate_texts(model, tokenizer, harmful_val))
            md_cell = sweep["cells"].get(f"L{layer}_a{mag}")
            cells[f"L{layer}_a{mag}"] = {
                "layer": layer, "alpha": -mag,
                "caa_harmful_val_refusal": r_harmful,
                "meandiff_harmful_val_refusal": (
                    md_cell["harmful_val_refusal"] if md_cell else None),
                "meandiff_ppl_delta_pct": (
                    md_cell["ppl_delta_pct"] if md_cell else None),
            }
            print(f"  L{layer} a-{mag}: caa {r_harmful:.3f} vs "
                  f"meandiff {cells[f'L{layer}_a{mag}']['meandiff_harmful_val_refusal']}")

    # harmless refusal at CAA's best cell
    best_key = min(cells, key=lambda k: cells[k]["caa_harmful_val_refusal"])
    best = cells[best_key]
    with steer(model, caa[best["layer"]], layer=best["layer"], alpha=best["alpha"]):
        best["caa_harmless_refusal"] = refusal_rate(
            generate_texts(model, tokenizer, harmless))
    print(f"best CAA cell {best_key}: harmless refusal "
          f"{best['caa_harmless_refusal']:.3f}")
    report["spotcheck"] = {"layers": top, "alphas": [-a for a in SPOTCHECK_ALPHAS],
                           "cells": cells, "best_cell": best_key}
    OUT_JSON.write_text(json.dumps(report, indent=2))

    # plot: cosine per layer + refusal vs |alpha| at best layer
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    layers_all = sorted(map(int, cos))
    axes[0].plot(layers_all, [cos[str(l)] for l in layers_all], marker="o", ms=3)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("cosine similarity")
    axes[0].set_title("v_caa vs v_meandiff alignment")
    axes[0].axhline(0, ls="--", c="gray", lw=0.8)
    l0 = best["layer"]
    xs = SPOTCHECK_ALPHAS
    axes[1].plot(xs, [cells[f"L{l0}_a{a}"]["caa_harmful_val_refusal"] for a in xs],
                 marker="o", label="CAA pairs")
    axes[1].plot(xs, [cells[f"L{l0}_a{a}"]["meandiff_harmful_val_refusal"] for a in xs],
                 marker="s", label="mean-diff")
    axes[1].set_xlabel("|alpha| (jailbreak sign)")
    axes[1].set_ylabel("harmful_val refusal rate")
    axes[1].set_title(f"effectiveness at layer {l0}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "caa_vs_meandiff.png", dpi=150)
    print(f"saved {RESULTS_DIR / 'caa_vs_meandiff.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pairs", "extract", "spotcheck"])
    ap.add_argument("--gen-alpha", type=float, default=GEN_ALPHA,
                    help="compliance-generation steering strength (negative)")
    args = ap.parse_args()

    model, tokenizer = load_model()
    if args.stage == "pairs":
        md_dirs = torch.load(MD_PT, weights_only=True)
        harmful_train = [r["instruction"]
                         for r in load_jsonl(DATA_DIR / "harmful_train.jsonl")]
        stage_pairs(model, tokenizer, md_dirs, harmful_train, args.gen_alpha)
    elif args.stage == "extract":
        stage_extract(model, tokenizer)
    else:
        stage_spotcheck(model, tokenizer)


if __name__ == "__main__":
    main()
