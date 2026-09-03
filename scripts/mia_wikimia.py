"""Membership-inference on WikiMIA with the CORRECT metric: TPR @ low FPR
(AUC only for reference). Implements loss, zlib, Min-K% Prob (arXiv:2310.16789)
and Min-K%++ (arXiv:2404.02936). Optionally applies a BLADE weight edit (scale a
selected edge mask by `factor`) to test whether editing membership-related weights
changes leakage.

Usage:
  BLADE_MODEL=... python scripts/mia_wikimia.py                 # baseline
  (edit hooks added by mia_defend.py for the defense sweep)
"""
import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch

from ttsafety.models import env_info, load_model

MODEL_ID = os.environ.get("BLADE_MODEL", "EleutherAI/pythia-2.8b")
SPLIT = os.environ.get("WIKIMIA_SPLIT", "WikiMIA_length64")
KFRAC = 0.2  # Min-K% uses the lowest 20% tokens
RESULTS = Path("results")


@torch.no_grad()
def token_logprobs(model, tok, text):
    """Return per-token logprob of the actual tokens, plus per-position mu/sigma
    of the model's next-token log-prob distribution (for Min-K%++)."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
    if ids.shape[1] < 2:
        return None
    logits = model(ids).logits[0].float()          # [T, V]
    logp = torch.log_softmax(logits, dim=-1)        # [T, V]
    tgt = ids[0, 1:]                                 # next tokens
    lp = logp[:-1]                                   # predict positions 0..T-2
    actual = lp[torch.arange(lp.shape[0]), tgt]      # [T-1]
    probs = lp.exp()
    mu = (probs * lp).sum(-1)                         # E[log p]
    var = (probs * lp.pow(2)).sum(-1) - mu.pow(2)
    sigma = var.clamp_min(1e-8).sqrt()
    return actual.cpu().numpy(), mu[torch.arange(lp.shape[0])].cpu().numpy(), \
        sigma[torch.arange(lp.shape[0])].cpu().numpy(), text


def scores_for(actual, mu, sigma, text):
    """Higher score => more likely MEMBER, for each method."""
    n = len(actual)
    k = max(1, int(n * KFRAC))
    loss = actual.mean()                             # mean logprob (higher=member)
    mink = np.sort(actual)[:k].mean()                # Min-K% Prob: lowest-k logprobs
    z = (actual - mu) / sigma
    minkpp = np.sort(z)[:k].mean()                   # Min-K%++: normalized
    zlib_e = len(zlib.compress(text.encode("utf-8")))
    zlib_s = loss / zlib_e * 1000                    # loss/zlib (higher=member)
    return {"loss": float(loss), "mink": float(mink),
            "minkpp": float(minkpp), "zlib": float(zlib_s)}


def tpr_at_fpr(scores, labels, fpr_target):
    """labels: 1=member. threshold set on NON-members at target FPR; TPR on members."""
    s = np.asarray(scores); y = np.asarray(labels)
    non = s[y == 0]; mem = s[y == 1]
    thr = np.quantile(non, 1 - fpr_target)           # FPR = frac non-members above thr
    return float((mem >= thr).mean())


def auc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(s))
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def evaluate(model, tok, rows):
    per = {m: [] for m in ("loss", "mink", "minkpp", "zlib")}
    labels = []
    for r in rows:
        out = token_logprobs(model, tok, r["input"])
        if out is None:
            continue
        sc = scores_for(*out)
        for m in per:
            per[m].append(sc[m])
        labels.append(r["label"])
    report = {}
    for m in per:
        report[m] = {"auc": auc(per[m], labels),
                     "tpr@1%fpr": tpr_at_fpr(per[m], labels, 0.01),
                     "tpr@5%fpr": tpr_at_fpr(per[m], labels, 0.05)}
    return report, per, labels


def main():
    from datasets import load_dataset
    rows = list(load_dataset("swj0419/WikiMIA", split=SPLIT))
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"{MODEL_ID} | {SPLIT} | n={len(rows)}", flush=True)
    report, _, labels = evaluate(model, tok, rows)
    n_mem = int(np.sum(labels)); n_non = len(labels) - n_mem
    print(f"members={n_mem} non-members={n_non}", flush=True)
    print(f"{'method':8} {'AUC':>6} {'TPR@1%FPR':>10} {'TPR@5%FPR':>10}", flush=True)
    for m, v in report.items():
        print(f"{m:8} {v['auc']:>6.3f} {v['tpr@1%fpr']:>10.3f} {v['tpr@5%fpr']:>10.3f}",
              flush=True)
    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"mia_baseline_{tag}.json").write_text(json.dumps(
        {"model": MODEL_ID, "split": SPLIT, "n_member": n_mem, "n_nonmember": n_non,
         "report": report, "env": env_info()}, indent=2))
    print(f"saved results/mia_baseline_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
