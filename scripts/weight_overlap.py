"""Overlap between sycophancy weights and refusal ("jailbreaking") weights.

Both are BLADE (signed-actdiff-edge) scores over the SAME target pool (L7-L18
o_proj+down_proj, 415,236,096 weights). We compare the top-scoring selections:
  * intersection / Jaccard / overlap-coefficient at matched k
  * enrichment vs chance (observed overlap / expected-if-independent)
  * per-layer overlap
  * continuous cosine similarity of the raw score vectors (threshold-free)

Refusal scores: data/weight_scores/edge.pt (existing).
Sycophancy scores: data/weight_scores/sycophancy_edge.pt (recomputed+cached here).
Outputs: results/weight_overlap.json + results/weight_overlap.png
"""

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCORES = DATA / "weight_scores"
REFUSAL_PT = SCORES / "edge.pt"
SYCO_PT = SCORES / "sycophancy_edge.pt"
LAYERS = list(range(7, 19))
COMPONENTS = "both"
K_REFUSAL = 207_618      # refusal headline (0.05% of pool)
K_SYCO = 2_076_180       # sycophancy headline (0.5% of pool)


def ensure_syco_scores():
    """Recompute + cache sycophancy edge scores if not on disk."""
    if SYCO_PT.exists():
        return
    from ttsafety.models import load_model
    from ttsafety.sycophancy import (collect_input_moments,
                                     extract_sycophancy_direction,
                                     fetch_sycophancy, make_splits, score_edges)
    print("recomputing sycophancy edge scores ...", flush=True)
    model, tok = load_model()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = fetch_sycophancy(DATA / "sycophancy")
    train = make_splits(rows)["train"]
    directions = extract_sycophancy_direction(model, tok, train)
    mu_s = collect_input_moments(model, tok, [r["biased"] for r in train],
                                 LAYERS, COMPONENTS)
    mu_n = collect_input_moments(model, tok, [r["neutral"] for r in train],
                                 LAYERS, COMPONENTS)
    scores = score_edges(model, directions, mu_s, mu_n, LAYERS, COMPONENTS)
    torch.save({"scores": scores}, SYCO_PT)
    print(f"saved {SYCO_PT}", flush=True)
    del model
    torch.cuda.empty_cache()


def load_scores(path):
    return torch.load(path, map_location="cpu", weights_only=False)["scores"]


def global_offsets(scores):
    """Assign each matrix a base offset in a single global weight index space."""
    off, cur = {}, 0
    for name in sorted(scores):
        off[name] = cur
        cur += scores[name].numel()
    return off, cur


def topk_global(scores, k, offsets, per_matrix_cap=0.10):
    """Global top-k flat indices (with per-matrix 10% cap), in global index space."""
    vals, gidx = [], []
    for name in sorted(scores):
        flat = scores[name].float().flatten()
        cap = max(1, int(per_matrix_cap * flat.numel()))
        v, loc = torch.topk(flat, cap, largest=True, sorted=False)
        vals.append(v)
        gidx.append(loc.long() + offsets[name])
    vals = torch.cat(vals)
    gidx = torch.cat(gidx)
    order = torch.topk(vals, k, largest=True, sorted=False).indices
    return gidx[order]


def per_layer_counts(sel_global, offsets, scores):
    """Map selected global indices back to layer indices."""
    bounds = sorted((off, name) for name, off in offsets.items())
    counts = {}
    sel_sorted = sel_global
    for name in scores:
        lo = offsets[name]
        hi = lo + scores[name].numel()
        layer = int(name.split(".")[1])
        n = int(((sel_global >= lo) & (sel_global < hi)).sum())
        counts[layer] = counts.get(layer, 0) + n
    return counts


def main():
    ensure_syco_scores()
    refusal = load_scores(REFUSAL_PT)
    syco = load_scores(SYCO_PT)
    assert set(refusal) == set(syco), "score matrices must match"
    offsets, pool = global_offsets(refusal)
    print(f"pool = {pool:,} weights", flush=True)

    report = {"pool": pool, "k_refusal": K_REFUSAL, "k_syco": K_SYCO,
              "layers": LAYERS, "comparisons": {}}

    def compare(name, k_ref, k_syc):
        r = topk_global(refusal, k_ref, offsets)
        s = topk_global(syco, k_syc, offsets)
        rset = set(r.tolist())
        inter = len(rset.intersection(s.tolist()))
        union = len(rset) + len(s) - inter
        jacc = inter / union
        ov_coeff = inter / min(len(r), len(s))
        expected = k_ref * k_syc / pool          # if independent
        enrich = inter / expected if expected else float("inf")
        return {"k_refusal": k_ref, "k_syco": k_syc, "intersection": inter,
                "jaccard": jacc, "overlap_coeff": ov_coeff,
                "expected_if_independent": expected, "enrichment": enrich}, r, s

    # (a) matched size at refusal headline k
    report["comparisons"]["matched_k=207618"], r_m, s_m = compare(
        "matched", K_REFUSAL, K_REFUSAL)
    # (b) each at its own headline
    report["comparisons"]["headline_refusal_vs_syco"], r_h, s_h = compare(
        "headline", K_REFUSAL, K_SYCO)

    # per-layer overlap at matched k
    r_layers = per_layer_counts(r_m, offsets, refusal)
    s_layers = per_layer_counts(s_m, offsets, syco)
    inter_set = set(r_m.tolist()).intersection(s_m.tolist())
    inter_t = torch.tensor(sorted(inter_set)) if inter_set else torch.tensor([], dtype=torch.long)
    o_layers = per_layer_counts(inter_t, offsets, refusal) if len(inter_t) else {}
    report["per_layer_matched"] = {
        str(l): {"refusal": r_layers.get(l, 0), "syco": s_layers.get(l, 0),
                 "overlap": o_layers.get(l, 0)} for l in LAYERS}

    # continuous cosine similarity of raw score vectors (streamed per matrix)
    dot = rn = sn = 0.0
    for name in sorted(refusal):
        a = refusal[name].float().flatten()
        b = syco[name].float().flatten()
        dot += float((a * b).sum())
        rn += float((a * a).sum())
        sn += float((b * b).sum())
    report["score_cosine"] = dot / (rn ** 0.5 * sn ** 0.5)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "weight_overlap.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== overlap ===", flush=True)
    for name, c in report["comparisons"].items():
        print(f"{name}: inter={c['intersection']:,} jaccard={c['jaccard']:.4f} "
              f"overlap_coeff={c['overlap_coeff']:.4f} "
              f"enrichment={c['enrichment']:.1f}x (vs chance)", flush=True)
    print(f"score cosine (threshold-free): {report['score_cosine']:.4f}", flush=True)
    print("saved results/weight_overlap.json", flush=True)
    make_plot(report)


def make_plot(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})", flush=True)
        return
    pl = report["per_layer_matched"]
    layers = [int(l) for l in pl]
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.38
    ax.bar([l - w / 2 for l in layers], [pl[str(l)]["refusal"] for l in layers],
           w, label="refusal top-207k", color="#3d5a80")
    ax.bar([l + w / 2 for l in layers], [pl[str(l)]["syco"] for l in layers],
           w, label="sycophancy top-207k", color="#e07a5f")
    ax.bar(layers, [pl[str(l)]["overlap"] for l in layers], w * 0.5,
           label="overlap", color="#3a1f14")
    ax.set_xlabel("layer")
    ax.set_ylabel("selected weights")
    c = report["comparisons"]["matched_k=207618"]
    ax.set_title(f"Refusal vs sycophancy weights (matched top-207k)\n"
                 f"Jaccard {c['jaccard']:.3f} · overlap-coeff {c['overlap_coeff']:.3f} "
                 f"· {c['enrichment']:.0f}× vs chance · score-cosine {report['score_cosine']:.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "weight_overlap.png", dpi=130)
    print("saved results/weight_overlap.png", flush=True)


if __name__ == "__main__":
    main()
