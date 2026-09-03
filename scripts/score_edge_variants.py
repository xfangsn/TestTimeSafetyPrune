"""Edge-score improvement variants (tasks #2 and #3).

Baseline edge score: s_ij = max(r_i * W_ij * da_j, 0) with r the per-layer
unit refusal direction and da_j = mean(harmful) - mean(harmless) writer-input
activation at the last non-pad prompt token (scripts/score_refusal_weights.py).

Variants:
  edge_signcons (task #2): per-sample contributions c_ij(x) = r_i*W_ij*da_j(x)
    with da_j(x) = a_j(x) - mu^U_j (harmless mean). Accumulate per edge
    n_pos = #{harmful x: c>0} and sum_pos = sum of positive c.
    score = (sum_pos/256) * (n_pos/256)^gamma, gamma=1. The pure consistency
    score (n_pos/256) is saved separately as a diagnostic.
  edge_trimmed (task #2): da from 10%-trimmed per-neuron means (both tails,
    harmful 256 and harmless 320 trimmed separately).
  edge_subspace_k (task #3): r replaced by the sum of the top-k PCA
    components of harmful last-token residual activations centered by the
    harmless mean; score = max(sum_k' r^k'_i * W_ij * da_j, 0) (sum then
    clamp). Clamp-then-sum is computed as a diagnostic (top-k overlap only).

All scores cached to data/weight_scores/. Activations are collected once and
cached (per-sample, last non-pad token, layers 7-18 residual writers +
block outputs).
"""

import argparse
import json
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.weight_edit import iter_residual_writers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = DATA / "weight_scores"
LAYERS = list(range(7, 19))
COMPONENTS = "both"
PROMPT_BATCH = 8
TRIM = 0.10
SUBSPACE_KS = (2, 4, 8)
GAMMA = 1.0


def save_scores(name: str, scores: dict, metadata: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.pt"
    torch.save({"scores": scores, "metadata": metadata}, path)
    summary = {
        "path": str(path.relative_to(ROOT)),
        "metadata": metadata,
        "matrices": {
            key: {
                "shape": list(value.shape),
                "positive_fraction": float((value > 0).float().mean()),
                "max": float(value.max()),
                "mean": float(value.float().mean()),
            }
            for key, value in scores.items()
        },
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {path}", flush=True)


@torch.no_grad()
def collect_per_sample(model, tokenizer, instructions):
    """Per-sample last-non-pad-token writer inputs + block-output residuals."""
    writers = dict(iter_residual_writers(model, LAYERS, COMPONENTS))
    blocks = list(get_decoder_layers(model))
    target_blocks = {l: blocks[l] for l in LAYERS}
    state = {}
    store_in = {name: [] for name in writers}
    store_resid = {l: [] for l in LAYERS}

    def make_pre(name):
        def hook(_module, args):
            values = args[0].float()
            rows = torch.arange(values.shape[0], device=values.device)
            store_in[name].append(values[rows, state["last_idx"]].cpu())
        return hook

    def make_post(layer):
        def hook(_module, _args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            rows = torch.arange(h.shape[0], device=h.device)
            store_resid[layer].append(
                h[rows, state["last_idx"]].float().cpu())
        return hook

    handles = [m.register_forward_pre_hook(make_pre(n))
               for n, m in writers.items()]
    handles += [b.register_forward_hook(make_post(l))
                for l, b in target_blocks.items()]
    try:
        for start in range(0, len(instructions), PROMPT_BATCH):
            texts = [chat_wrap(tokenizer, v)
                     for v in instructions[start:start + PROMPT_BATCH]]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            padding_side="right",
                            add_special_tokens=False).to(model.device)
            state["last_idx"] = enc["attention_mask"].sum(1) - 1
            model(**enc, use_cache=False)
            if (start // PROMPT_BATCH) % 10 == 0:
                print(f"  collect {start}/{len(instructions)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    inputs = {n: torch.cat(v) for n, v in store_in.items()}
    resid = {l: torch.cat(v) for l, v in store_resid.items()}
    return inputs, resid


def trimmed_mean(x: torch.Tensor, trim: float) -> torch.Tensor:
    """Per-column mean after dropping `trim` from both tails (dim 0)."""
    n = x.shape[0]
    k = int(n * trim)
    xs = x.sort(dim=0).values
    return xs[k:n - k].mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true",
                    help="collect and cache per-sample activations")
    ap.add_argument("--score", action="store_true",
                    help="compute variant scores from cached activations")
    args = ap.parse_args()
    if not (args.collect or args.score):
        ap.error("--collect and/or --score required")

    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]

    if args.collect:
        model, tokenizer = load_model()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for tag, insts in (("harmful", harmful), ("harmless", harmless)):
            print(f"collecting {tag} per-sample activations ...", flush=True)
            inputs, resid = collect_per_sample(model, tokenizer, insts)
            torch.save({"inputs": inputs,
                        "resid": {str(k): v for k, v in resid.items()},
                        "n": len(insts), "env": env_info()},
                       OUT_DIR / f"per_sample_{tag}.pt")
            print(f"saved per_sample_{tag}.pt", flush=True)

    if not args.score:
        return

    h = torch.load(OUT_DIR / "per_sample_harmful.pt", map_location="cpu",
                   weights_only=False)
    u = torch.load(OUT_DIR / "per_sample_harmless.pt", map_location="cpu",
                   weights_only=False)
    directions = torch.load(
        DATA / "directions" / "refusal_llama32_3b_instruct.pt",
        map_location="cpu", weights_only=True)
    model, tokenizer = load_model()
    writers = dict(iter_residual_writers(model, LAYERS, COMPONENTS))
    device = model.device
    names = sorted(writers)

    # ---- task #2: edge_signcons + consistency + edge_trimmed ----
    print("computing edge_signcons / consistency / edge_trimmed ...", flush=True)
    signcons, consistency, trimmed = {}, {}, {}
    signcons_stats = []
    for name in names:
        layer = int(name.split(".")[1])
        r = directions[layer].float()
        r = (r / r.norm()).to(device)
        W = writers[name].weight.detach().float()
        A_h = h["inputs"][name].to(device)          # (256, in)
        mu_u = u["inputs"][name].mean(0).to(device)  # (in,)
        M = r[:, None] * W                           # (out, in)
        n_pos = torch.zeros_like(M, dtype=torch.int32)
        sum_pos = torch.zeros_like(M)
        for i in range(A_h.shape[0]):
            c = M * (A_h[i] - mu_u)[None, :]
            pos = c > 0
            n_pos += pos
            sum_pos += c.clamp_min_(0)
        n = A_h.shape[0]
        cons = n_pos.float() / n
        score = (sum_pos / n) * cons.pow(GAMMA)
        signcons[name] = score.cpu().to(torch.float16)
        consistency[name] = cons.cpu().to(torch.float16)
        pos_frac = float((cons > 0).float().mean())
        import numpy as np
        pos_vals = cons[cons > 0].cpu().numpy()
        signcons_stats.append({
            "matrix": name,
            "edges_with_any_positive": pos_frac,
            "consistency_quantiles_on_positive": [
                float(np.quantile(pos_vals, q)) for q in (0.5, 0.9, 0.99)],
        })
        # trimmed means
        mu_h_t = trimmed_mean(h["inputs"][name].float(), TRIM).to(device)
        mu_u_t = trimmed_mean(u["inputs"][name].float(), TRIM).to(device)
        delta_t = mu_h_t - mu_u_t
        trimmed[name] = (M * delta_t[None, :]).clamp_min_(0).cpu().to(torch.float16)
    meta = {"layers": LAYERS, "components": COMPONENTS,
            "n_harmful": len(harmful), "n_harmless": len(harmless),
            "env": env_info()}
    save_scores("edge_signcons", signcons,
                {**meta, "score": "edge_signcons", "gamma": GAMMA})
    save_scores("edge_signcons_consistency", consistency,
                {**meta, "score": "edge_signcons_consistency",
                 "note": "diagnostic: pure n_pos/n sign consistency"})
    save_scores("edge_trimmed", trimmed,
                {**meta, "score": "edge_trimmed", "trim": TRIM})
    (OUT_DIR / "edge_signcons_stats.json").write_text(json.dumps({
        "gamma": GAMMA, "per_matrix": signcons_stats}, indent=2))

    # ---- task #3: PCA subspace directions + edge_subspace_k ----
    print("computing PCA subspace directions ...", flush=True)
    pca_dirs, pca_stats = {}, []
    for layer in LAYERS:
        X = h["resid"][str(layer)].float()           # (256, 3072)
        mu_u = u["resid"][str(layer)].mean(0)
        Xc = X - mu_u[None, :]                        # center by harmless mean
        cov = Xc.T @ Xc / Xc.shape[0]
        evals, evecs = torch.linalg.eigh(cov)
        order = evals.argsort(descending=True)
        evals, evecs = evals[order], evecs[:, order]
        comps = evecs[:, :SUBSPACE_KS[-1]].T.contiguous()  # (8, 3072) rows unit
        pca_dirs[layer] = comps
        vd = directions[layer].float()
        vd = vd / vd.norm()
        total = evals.clamp_min(0).sum()
        pca_stats.append({
            "layer": layer,
            "cos_pc1_refusal_direction": float((comps[0] * vd).sum().abs()),
            "explained_variance_top8": [
                float(evals[i].clamp_min(0) / total) for i in range(8)],
        })
    torch.save({"directions": pca_dirs, "stats": pca_stats},
               OUT_DIR / "pca_subspace_directions.pt")
    (OUT_DIR / "pca_subspace_directions.json").write_text(json.dumps(
        {"stats": pca_stats,
         "note": "PCs of harmful last-token residual centered by harmless "
                 "mean; rows are unit vectors"}, indent=2))
    print("PC1 cosine with cached refusal direction per layer:",
          {s["layer"]: round(s["cos_pc1_refusal_direction"], 3)
           for s in pca_stats}, flush=True)

    for k in SUBSPACE_KS:
        scores, clampfirst = {}, {}
        for name in names:
            layer = int(name.split(".")[1])
            comps = pca_dirs[layer][:k].to(device)      # (k, 3072)
            W = writers[name].weight.detach().float()
            A_h = h["inputs"][name].to(device)
            delta = (A_h.mean(0) - u["inputs"][name].float()
                     .mean(0).to(device))
            r_sum = comps.sum(0)
            scores[name] = (r_sum[:, None] * W
                            * delta[None, :]).clamp_min_(0).cpu().to(torch.float16)
            cf = torch.zeros_like(W)
            for j in range(k):
                cf += (comps[j][:, None] * W * delta[None, :]).clamp_min_(0)
            clampfirst[name] = cf.cpu().to(torch.float16)
        save_scores(f"edge_subspace_k{k}", scores,
                    {**meta, "score": f"edge_subspace_k{k}", "k": k,
                     "combination": "sum-then-clamp"})
        save_scores(f"edge_subspace_k{k}_clampfirst", clampfirst,
                    {**meta, "score": f"edge_subspace_k{k}_clampfirst",
                     "k": k, "combination": "clamp-then-sum (diagnostic)"})
    print("done.", flush=True)


if __name__ == "__main__":
    main()
