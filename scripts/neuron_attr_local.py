"""N6a: injection-layer-local neuron attribution.

Injection layer l := inject v̂_{l-1} at block l's INPUT (hook on block l-1's
output — the old "L{l-1}" convention). Measurement = layer l's 8192 MLP
post-SwiGLU neurons only.

Per injection layer: fresh fixed-alpha (-2) generations on harmful_train,
flip = refused at baseline (from flip_alphas.jsonl) AND non-refusal at
alpha=-2. Attribution (teacher-forced, same completion, inject on/off):
  D1[n] = Σ_flipped |softmax(x1)[n] - softmax(x2)[n]|
  D2[n] = Σ_flipped log clamp(softmax(x1)[n], eps=1e-12)
Main layer 9 uses its own flipped subset; other layers reuse the layer-9
flipped subset (caveat recorded in results) with their own completions.

Chunkable: --layers 9 / --layers 10,12,14 / ... ; --finalize merges and
writes results/neuron_attr_local.json + figures.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ttsafety.data import load_jsonl
from ttsafety.hooks import capture_neurons
from ttsafety.generate import generate_texts
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.steer import steer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_TAG = "llama32_3b_instruct"
LAYERS_ALL = [9, 10, 12, 14, 16, 18]
MAIN_LAYER = 9
ALPHA = -2.0
MAX_NEW_TOKENS = 128
EPS = 1e-12


def gen_path(layer):
    return DATA_DIR / f"gen_alpha2_inject{layer}.jsonl"


def partial_path(layer):
    return DATA_DIR / f"neuron_importance_local_{layer}.pt"


def response_span_start(tokenizer, prompt, full):
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    m = 0
    for a, b in zip(p_ids, f_ids):
        if a != b:
            break
        m += 1
    if m >= len(f_ids):
        raise ValueError("empty response span")
    return m


def layer_response_mean(model, tokenizer, prompt, completion, layer,
                        inject_vec=None):
    """Mean post-SwiGLU activations at `layer` over the response span."""
    full = prompt + completion + "<|eot_id|>"
    start = response_span_start(tokenizer, prompt, full)
    enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)

    def _fwd():
        caps = capture_neurons(model, enc["input_ids"], enc["attention_mask"],
                               layers=[layer])
        return caps["mlp"][layer][0, start:].mean(dim=0)

    if inject_vec is None:
        return _fwd()
    with steer(model, inject_vec, layer=layer - 1, alpha=ALPHA):
        return _fwd()


def run_layer(layer):
    model, tokenizer = load_model()
    directions = torch.load(
        DATA_DIR / "directions" / f"refusal_{MODEL_TAG}.pt", weights_only=True)
    vec = directions[layer - 1]
    train = load_jsonl(DATA_DIR / "harmful_train.jsonl")
    flip_records = {r["idx"]: r for r in load_jsonl(DATA_DIR / "flip_alphas.jsonl")}

    # fresh fixed-alpha generations at this injection layer
    if gen_path(layer).exists():
        gens = {r["idx"]: r for r in load_jsonl(gen_path(layer))}
        print(f"reusing existing generations for layer {layer}")
    else:
        print(f"generating at alpha={ALPHA}, injection layer {layer} "
              f"(hook block {layer - 1} output) ...")
        instructions = [r["instruction"] for r in train]
        with steer(model, vec, layer=layer - 1, alpha=ALPHA):
            outs = generate_texts(model, tokenizer, instructions, MAX_NEW_TOKENS)
        gens = {}
        for idx, (rec, out) in enumerate(zip(train, outs)):
            flipped = (flip_records[idx]["refused_baseline"]
                       and not is_refusal(out))
            gens[idx] = {"idx": idx, "instruction": rec["instruction"],
                         "output": out, "flipped": flipped}
        with gen_path(layer).open("w", encoding="utf-8") as f:
            for idx in sorted(gens):
                f.write(json.dumps(gens[idx], ensure_ascii=False) + "\n")
    flip_rate = sum(r["flipped"] for r in gens.values()) / len(gens)
    print(f"layer {layer}: flip rate at alpha={ALPHA} = {flip_rate:.3f}")

    # attribution subset: layer 9 uses its own flips; others reuse layer 9's
    if layer == MAIN_LAYER:
        subset = [r for r in gens.values() if r["flipped"]]
        subset_from = "own"
    else:
        main_gens = load_jsonl(gen_path(MAIN_LAYER))
        subset_idx = {r["idx"] for r in main_gens if r["flipped"]}
        subset = [gens[i] for i in sorted(subset_idx)]
        subset_from = f"layer{MAIN_LAYER}"
    print(f"attribution subset: {len(subset)} samples (from {subset_from})")

    d1 = torch.zeros(8192)
    d2 = torch.zeros(8192)
    for n, rec in enumerate(subset):
        prompt = chat_wrap(tokenizer, rec["instruction"])
        x1 = layer_response_mean(model, tokenizer, prompt, rec["output"], layer)
        x2 = layer_response_mean(model, tokenizer, prompt, rec["output"], layer,
                                 inject_vec=vec)
        p1 = torch.softmax(x1, dim=0)
        p2 = torch.softmax(x2, dim=0)
        d1 += (p1 - p2).abs()
        d2 += torch.log(p1.clamp(min=EPS))
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(subset)}")
    torch.save({"layer": layer, "D1": d1, "D2": d2,
                "n_subset": len(subset), "flip_rate": flip_rate,
                "subset_from": subset_from}, partial_path(layer))
    print(f"saved {partial_path(layer)}")


def _rank(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x, y):
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def gini(v):
    v = np.sort(np.asarray(v, dtype=np.float64).ravel())
    if v.sum() == 0:
        return 0.0
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def concentration(v):
    """Top-1% share + Gini; v is shifted to non-negative (D2 is <= 0)."""
    v = np.asarray(v, dtype=np.float64)
    v = v - v.min()
    k1 = int(0.01 * len(v))
    top = np.sort(v)[::-1]
    return {"top1pct_share": float(top[:k1].sum() / v.sum()) if v.sum() else 0.0,
            "gini": gini(v)}


def finalize():
    merged = {}
    for layer in LAYERS_ALL:
        p = partial_path(layer)
        if not p.exists():
            raise SystemExit(f"missing {p} — run layer {layer} first")
        merged[layer] = torch.load(p, weights_only=True)
    torch.save({str(l): {"D1": m["D1"], "D2": m["D2"]}
                for l, m in merged.items()},
               DATA_DIR / "neuron_importance_local.pt")

    report = {"config": {"model": MODEL_TAG, "layers": LAYERS_ALL,
                         "main_layer": MAIN_LAYER, "alpha": ALPHA,
                         "max_new_tokens": MAX_NEW_TOKENS,
                         "caveat": "layers != 9 reuse the layer-9 flipped subset"},
              "env": env_info(), "per_layer": {}}
    for layer, m in merged.items():
        d1, d2 = m["D1"].numpy(), m["D2"].numpy()
        report["per_layer"][str(layer)] = {
            "flip_rate": m["flip_rate"],
            "n_subset": m["n_subset"],
            "subset_from": m["subset_from"],
            "spearman_D1_D2": spearman(d1, d2),
            "D1": concentration(d1),
            "D2": concentration(d2),
        }
        c = report["per_layer"][str(layer)]
        print(f"L{layer}: flip {c['flip_rate']:.3f} n={c['n_subset']} "
              f"spearman {c['spearman_D1_D2']:+.3f} | "
              f"D1 top1% {c['D1']['top1pct_share']:.3f} gini {c['D1']['gini']:.3f} | "
              f"D2 top1% {c['D2']['top1pct_share']:.3f} gini {c['D2']['gini']:.3f}")

    out = RESULTS_DIR / "neuron_attr_local.json"
    out.write_text(json.dumps(report, indent=2))

    # D1-vs-D2 scatter per layer (subsampled) + concentration bars
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (layer, m) in zip(axes.flat, merged.items()):
        d1, d2 = m["D1"].numpy(), m["D2"].numpy()
        idx = np.random.default_rng(0).choice(len(d1), 2000, replace=False)
        ax.scatter(d1[idx], d2[idx], s=2, alpha=0.4)
        ax.set_title(f"inject L{layer} (rho={report['per_layer'][str(layer)]['spearman_D1_D2']:+.2f})",
                     fontsize=9)
        ax.set_xlabel("D1"); ax.set_ylabel("D2")
    fig.suptitle("D1 (intervention change) vs D2 (refusal-state activation)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "neuron_attr_local_scatter.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LAYERS_ALL))
    w = 0.2
    for i, (metric, label) in enumerate(
            [("D1", "D1 top1%"), ("D2", "D2 top1%")]):
        vals = [report["per_layer"][str(l)][metric]["top1pct_share"]
                for l in LAYERS_ALL]
        ax.bar(x + i * w, vals, w, label=label)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([f"L{l}" for l in LAYERS_ALL])
    ax.set_ylabel("top-1% importance share")
    ax.set_title("Concentration per injection layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "neuron_attr_local_concentration.png", dpi=150)
    print(f"saved {out} + scatter + concentration figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
        return
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else LAYERS_ALL)
    if any(l != MAIN_LAYER for l in layers) and not gen_path(MAIN_LAYER).exists():
        raise SystemExit("run layer 9 first (others reuse its flipped subset)")
    for layer in layers:
        run_layer(layer)


if __name__ == "__main__":
    main()
