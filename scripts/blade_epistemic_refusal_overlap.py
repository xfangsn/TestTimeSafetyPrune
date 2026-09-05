"""Weight-overlap analysis: do the BLADE-G weights selected for EPISTEMIC uncertainty and for REFUSAL
coincide, or are they disjoint? (specificity check without behavioral cross-eval). Both scored on the
SAME layer set (all layers) so the comparison is fair; report top-rho selection overlap / Jaccard and
compare to the random-overlap expectation (~rho under matched per-matrix counts). Near-random overlap =>
the two behaviors live in different weights (supports 'unrelated').

Usage: BLADE_MODEL=Qwen/Qwen3-8B .venv/bin/python scripts/blade_epistemic_refusal_overlap.py
"""
import json
import os
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text
from ttsafety.extract import extract_refusal_direction
import ttsafety.extract as EX
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from ttsafety.weight_edit import iter_residual_writers
from blade_epistemic_els import qwen_wrap, last_token_moments, COMPONENTS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"; DATA = ROOT / "data"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")


def _medpos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def select(model, direction, muA, muB, layers, Q, rho):
    lam = _medpos(score_edges(model, direction, muA, muB, layers, COMPONENTS)) / _medpos(Q)
    S = score_edges_g(model, direction, muA, muB, layers, COMPONENTS, Q=Q, lam=lam, abstain=True)
    S = {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
    return selection_from_ranking(rank_weight_indices(S, max(0.03, rho)), rho)


def overlap(selA, selB):
    inter = total_a = total_b = 0
    per_layer = {}
    keys = set(selA) | set(selB)
    for k in keys:
        a = set(selA.get(k, torch.tensor([], dtype=torch.long)).tolist())
        b = set(selB.get(k, torch.tensor([], dtype=torch.long)).tolist())
        i = len(a & b)
        inter += i; total_a += len(a); total_b += len(b)
        if a or b:
            L = int(k.split(".")[1])
            per_layer.setdefault(L, [0, 0, 0])
            per_layer[L][0] += i; per_layer[L][1] += len(a); per_layer[L][2] += len(b)
    union = total_a + total_b - inter
    return {"intersection": inter, "n_epistemic": total_a, "n_refusal": total_b,
            "jaccard": inter / max(union, 1), "per_layer": per_layer}


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))
    total_entries = None

    # epistemic
    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    unc = [r["question"] for r in rows if r["label"] == 1]
    cert = [r["question"] for r in rows if r["label"] == 0]
    dir_e = extract_refusal_direction(model, tok, unc, cert)
    muUNC = last_token_moments(model, tok, unc, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert, all_layers, COMPONENTS, qwen_wrap)

    # refusal (same construction: harmful vs harmless, last prompt token)
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    dir_r = extract_refusal_direction(model, tok, harmful, harmless)
    muH = last_token_moments(model, tok, harmful, all_layers, COMPONENTS, qwen_wrap)
    muU = last_token_moments(model, tok, harmless, all_layers, COMPONENTS, qwen_wrap)

    print("Q ...", flush=True)
    Q, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=load_c4_text(),
                                         seqlen=2048, batch_size=2, mode="g1scalar", max_tokens=65536)
    total_entries = int(sum(m.weight.numel() for _, m in iter_residual_writers(model, all_layers, COMPONENTS)))

    report = {"model": MODEL_ID, "layers": "all", "total_writer_entries": total_entries,
              "env": env_info(), "by_rho": []}
    print(f"total residual-writer entries (all layers): {total_entries}", flush=True)
    for rho in (0.002, 0.005, 0.02):
        selE = select(model, dir_e, muUNC, muCERT, all_layers, Q, rho)
        selR = select(model, dir_r, muH, muU, all_layers, Q, rho)
        ov = overlap(selE, selR)
        # random expectation: two independent rho-fraction selections of N entries -> E[|A&B|] ~ rho^2 * N
        exp_rand = (ov["n_epistemic"] / total_entries) * ov["n_refusal"]
        fold = ov["intersection"] / max(exp_rand, 1e-9)
        row = {"rho": rho, **{k: ov[k] for k in ("intersection", "n_epistemic", "n_refusal", "jaccard")},
               "expected_random_intersection": round(exp_rand, 1),
               "overlap_vs_random_fold": round(fold, 2)}
        report["by_rho"].append(row)
        print(f"  rho={rho:.1%}: inter={ov['intersection']} epi={ov['n_epistemic']} ref={ov['n_refusal']} "
              f"Jaccard={ov['jaccard']:.4f}  exp_random={exp_rand:.1f}  x_random={fold:.2f}", flush=True)
        # top overlapping layers
        pl = sorted(ov["per_layer"].items(), key=lambda kv: -kv[1][0])[:5]
        print("    top-overlap layers (L: inter/epi/ref): " +
              " ".join(f"L{L}:{v[0]}/{v[1]}/{v[2]}" for L, v in pl if v[0] > 0), flush=True)

    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"epistemic_refusal_overlap_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/epistemic_refusal_overlap_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
