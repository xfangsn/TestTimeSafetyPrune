"""N2: neuron importance from steered vs unsteered teacher-forced forwards.

For each flipped sample (data/flip_alphas.jsonl, alpha_star not None):
  full text = chat_wrap(instruction) + steered completion + <|eot_id|>
  x_l = mean over RESPONSE token positions of post-SwiGLU MLP activations
Variants per layer (x1 = no steering, x2 = steering at L8 with alpha*):
  softmax: importance += |softmax(x1) - softmax(x2)|          (main)
  l1:      importance += |x1/||x1||_1 - x2/||x2||_1|          (robustness)
  z:       importance += |zscore(x1 - x2)|  (z over the 8192 dims; robustness)
  resid_l1: same as l1 but on the residual stream (auxiliary view, 3072-dim)
Also per-sample total softmax distance at alpha* and at a FIXED alpha=4
(de-confounds the mechanical alpha* -> distance relationship).

Chunked: `--start/--end` writes data/neuron_attr_partial_{start}_{end}.pt;
`--finalize` merges, runs analyses, writes results/neuron_importance.json,
data/neuron_importance.pt, and the PNGs.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ttsafety.data import load_jsonl
from ttsafety.hooks import capture_neurons
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.steer import steer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_TAG = "llama32_3b_instruct"
STEER_LAYER = 8
FIXED_ALPHA = 4.0
N_LAYERS = 28


def response_span_start(tokenizer, prompt: str, full: str) -> int:
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


def sample_response_means(model, tokenizer, vec, prompt, completion, alpha):
    """Per-layer response-mean neuron + resid activations, optionally steered."""
    full = prompt + completion + "<|eot_id|>"
    start = response_span_start(tokenizer, prompt, full)
    enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)

    def _capture():
        caps = capture_neurons(model, enc["input_ids"], enc["attention_mask"])
        mlp, resid = {}, {}
        for l in range(N_LAYERS):
            mlp[l] = caps["mlp"][l][0, start:].mean(dim=0)
            resid[l] = caps["resid"][l][0, start:].mean(dim=0)
        return mlp, resid

    if alpha == 0:
        return _capture()
    with steer(model, vec, layer=STEER_LAYER, alpha=-alpha):
        return _capture()


def run_range(start, end):
    model, tokenizer = load_model()
    directions = torch.load(
        DATA_DIR / "directions" / f"refusal_{MODEL_TAG}.pt", weights_only=True)
    vec = directions[STEER_LAYER]
    flipped = [r for r in load_jsonl(DATA_DIR / "flip_alphas.jsonl")
               if r["alpha_star"] is not None]
    chunk = flipped[start:end]
    print(f"{len(chunk)} flipped samples in [{start}, {end})")

    imp = {"softmax": torch.zeros(N_LAYERS, 8192),
           "l1": torch.zeros(N_LAYERS, 8192),
           "z": torch.zeros(N_LAYERS, 8192),
           "resid_l1": torch.zeros(N_LAYERS, 3072)}
    records = []
    for n, rec in enumerate(chunk):
        prompt = chat_wrap(tokenizer, rec["instruction"])
        x1, r1 = sample_response_means(model, tokenizer, vec, prompt,
                                       rec["output"], 0.0)
        x2, r2 = sample_response_means(model, tokenizer, vec, prompt,
                                       rec["output"], rec["alpha_star"])
        x3, _ = sample_response_means(model, tokenizer, vec, prompt,
                                      rec["output"], FIXED_ALPHA)
        d_star = d_fixed = 0.0
        for l in range(N_LAYERS):
            a, b, c = x1[l], x2[l], x3[l]
            d1 = torch.softmax(a, dim=0) - torch.softmax(b, dim=0)
            imp["softmax"][l] += d1.abs()
            imp["l1"][l] += (a / a.abs().sum() - b / b.abs().sum()).abs()
            delta = a - b
            std = delta.std()
            if std > 0:  # layers upstream of L8 have delta == 0 (negative control)
                imp["z"][l] += ((delta - delta.mean()) / std).abs()
            ra, rb = r1[l], r2[l]
            imp["resid_l1"][l] += (ra / ra.abs().sum()
                                   - rb / rb.abs().sum()).abs()
            d_star += d1.abs().sum().item()
            d_fixed += (torch.softmax(a, dim=0)
                        - torch.softmax(c, dim=0)).abs().sum().item()
        records.append({"idx": rec["idx"], "alpha_star": rec["alpha_star"],
                        "d_star_softmax": d_star, "d_fixed_softmax": d_fixed})
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(chunk)} done")
    out = DATA_DIR / f"neuron_attr_partial_{start}_{end}.pt"
    torch.save({"imp": imp, "records": records}, out)
    print(f"saved {out}")


# --- statistics helpers -----------------------------------------------------

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
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def finalize():
    partials = sorted(glob.glob(str(DATA_DIR / "neuron_attr_partial_*.pt")))
    if not partials:
        raise SystemExit("no partials found — run chunks first")
    imp = None
    records = []
    for p in partials:
        blob = torch.load(p, weights_only=False)
        if imp is None:
            imp = {k: v.clone() for k, v in blob["imp"].items()}
        else:
            for k in imp:
                imp[k] += blob["imp"][k]
        records.extend(blob["records"])
    print(f"merged {len(partials)} partials, {len(records)} samples")

    torch.save(imp, DATA_DIR / "neuron_importance.pt")

    alphas = [r["alpha_star"] for r in records]
    d_star = [r["d_star_softmax"] for r in records]
    d_fixed = [r["d_fixed_softmax"] for r in records]
    report = {
        "config": {"model": MODEL_TAG, "steer_layer": STEER_LAYER,
                   "fixed_alpha": FIXED_ALPHA, "n_samples": len(records)},
        "env": env_info(),
        "spearman_alpha_star_vs_d_at_alpha_star": spearman(alphas, d_star),
        "spearman_alpha_star_vs_d_fixed_alpha": spearman(alphas, d_fixed),
        "per_layer": {},
        "concentration": {},
        "negative_control": {},
    }

    for variant in ("softmax", "l1", "z"):
        m = imp[variant]
        layer_totals = m.sum(dim=1)
        total = layer_totals.sum().item()
        upstream = layer_totals[:STEER_LAYER].sum().item()
        flat = m.flatten()
        k1 = int(0.01 * flat.numel())
        k5 = int(0.05 * flat.numel())
        top = torch.topk(flat, k5).values  # descending
        top50_layers = torch.topk(m.flatten(), 50).indices // 8192
        report["per_layer"][variant] = {
            "totals": [round(v, 6) for v in (layer_totals / total).tolist()],
            "argmax_layer": int(layer_totals.argmax()),
        }
        report["concentration"][variant] = {
            "top1pct_share": float(top[:k1].sum() / total),
            "top5pct_share": float(top.sum() / total),
            "gini": gini(flat.numpy()),
            "top50_layer_hist": {
                str(l): int((top50_layers == l).sum())
                for l in range(N_LAYERS) if (top50_layers == l).any()},
        }
        report["negative_control"][variant] = {
            "L0_7_share": upstream / total,
            "L8plus_share": 1 - upstream / total,
        }

    out = RESULTS_DIR / "neuron_importance.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "env"}, indent=2))

    # plots
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant in ("softmax", "l1", "z"):
        ax.plot(report["per_layer"][variant]["totals"], marker="o", ms=3,
                label=f"mlp {variant}")
    resid_totals = imp["resid_l1"].sum(dim=1)
    ax.plot((resid_totals / resid_totals.sum()).tolist(), marker="s", ms=3,
            ls="--", label="resid l1")
    ax.axvline(STEER_LAYER - 0.5, ls=":", c="gray")
    ax.set_xlabel("layer")
    ax.set_ylabel("share of total importance")
    ax.set_title("Per-layer neuron importance (steering at L8)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "neuron_importance_layers.png", dpi=150)

    hist = report["concentration"]["softmax"]["top50_layer_hist"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([int(k) for k in hist], list(hist.values()))
    ax.set_xlabel("layer")
    ax.set_ylabel("count in top-50 neurons")
    ax.set_title("Top-50 neurons by importance: layer distribution (softmax)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "neuron_importance_top50_hist.png", dpi=150)
    print(f"saved {out} + layer curve + top50 hist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=248)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
    else:
        run_range(args.start, args.end)


if __name__ == "__main__":
    main()
