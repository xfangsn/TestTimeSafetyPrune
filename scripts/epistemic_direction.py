"""Scheme A step 2 — build the refusal-style epistemic-uncertainty direction and VALIDATE that it
is linearly decodable across question types (not memorising a template).

Direction:  v_l = mean act_l(uncertain) - mean act_l(certain)  at the LAST PROMPT TOKEN
            (Qwen3 chat template, enable_thinking=True, add_generation_prompt=True; pre-generation,
             fixed position) — the exact refusal construction (Arditi/CAA), just a different contrast.

Validation (leave-one-FAMILY-out CV, so train/test question types are disjoint -> no template leak):
  - per-layer logistic-probe AUROC on the held-out family (mean over folds);
  - diff-of-means projection AUROC on the held-out family (the v we actually edit with);
  - length baseline AUROC (prompt token count) — the direction must beat it.

Saves data/directions/epistemic_<tag>.pt = {layer: v_l fp32 CPU} and results/epistemic_direction_<tag>.json
Usage: .venv/bin/python scripts/epistemic_direction.py --model Qwen/Qwen3-8B --tag qwen3_8b
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ttsafety.hooks import capture_last_token
from ttsafety.models import load_model

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DIRS = ROOT / "data" / "directions"
THINKING = True   # set by --thinking; the WEIGHT edit uses thinking=OFF, so validate that regime


def qwen_wrap(tok, instr):
    m = [{"role": "user", "content": instr}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=THINKING)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--pairs", default="epistemic_pairs.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="chat-template enable_thinking; the WEIGHT edit uses off, so validate off too")
    args = ap.parse_args()
    global THINKING
    THINKING = args.thinking == "on"

    data = json.loads((RESULTS / args.pairs).read_text())
    rows = data["rows"]
    y = np.array([r["label"] for r in rows])           # 1 = uncertain
    fam = np.array([r["family"] for r in rows])
    texts = [r["question"] for r in rows]
    families = sorted(set(fam.tolist()))

    # SURFACE baseline (LOFO): word bag-of-words + prompt length -> logistic. The activation direction
    # must beat this to be more than lexical/length shortcut (codex: "length-controlled" was overstated).
    def surface_lofo(feat):
        aucs = []
        for held in families:
            tr, te = fam != held, fam == held
            if len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(feat[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(feat[te])))
        return float(np.mean(aucs))

    model, tok = load_model(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [qwen_wrap(tok, r["question"]) for r in rows]
    plen = np.array([len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts])

    print(f"capturing last-token acts for {len(prompts)} prompts ...", flush=True)
    acts = capture_last_token(model, tok, prompts, batch_size=args.batch_size)  # {layer:(N,H)}
    layers = sorted(acts)

    # length baseline (same LOFO protocol, trivial 1-D "probe" = threshold on length)
    len_auc = []
    for held in families:
        te = fam == held
        try:
            len_auc.append(roc_auc_score(y[te], plen[te]))
        except ValueError:
            pass
    len_auc = float(np.mean(len_auc))

    # word bag-of-words (LOFO -> new-family words unseen, so this is a fair "generalizable lexical" test)
    bow = CountVectorizer(min_df=2).fit_transform(texts).toarray().astype(float)
    bow_auc = surface_lofo(bow)
    lenfeat = plen.reshape(-1, 1).astype(float)
    len_logit_auc = surface_lofo(lenfeat)
    surf_auc = surface_lofo(np.hstack([bow, lenfeat]))
    print(f"[thinking={args.thinking}] SURFACE LOFO baselines: length {len_logit_auc:.3f} "
          f"bow {bow_auc:.3f} bow+len {surf_auc:.3f}", flush=True)

    report = {"model": args.model, "tag": args.tag, "n": len(rows), "thinking": args.thinking,
              "families": families, "length_baseline_auroc": round(len_auc, 3),
              "surface_lofo": {"length": round(len_logit_auc, 3), "bow": round(bow_auc, 3),
                               "bow_plus_len": round(surf_auc, 3)}, "per_layer": {}}
    per_layer = {}
    for L in layers:
        X = acts[L].numpy()
        probe_aucs, dom_aucs = [], []
        for held in families:
            tr, te = fam != held, fam == held
            if len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2:
                continue
            # logistic probe (standardised, L2)
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            probe_aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
            # diff-of-means direction on TRAIN families, project held-out family
            v = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
            dom_aucs.append(roc_auc_score(y[te], X[te] @ v))
        pa, da = float(np.mean(probe_aucs)), float(np.mean(dom_aucs))
        per_layer[L] = {"probe_auroc": round(pa, 3), "diffmeans_auroc": round(da, 3)}

    report["per_layer"] = {str(k): v for k, v in per_layer.items()}
    best_probe = max(per_layer, key=lambda L: per_layer[L]["probe_auroc"])
    best_dom = max(per_layer, key=lambda L: per_layer[L]["diffmeans_auroc"])

    # per-held-family diagnostic at best diff-means layer + a length-matched-family probe.
    # length is a confound (uncertain prompts run longer); event_year/person_attr are ~length-matched,
    # so a high held-out AUROC there = epistemic signal, not length. Also report length-AUROC per family.
    Lb = best_dom
    Xb = acts[Lb].numpy()
    per_fam = {}
    for held in families:
        tr, te = fam != held, fam == held
        if len(set(y[te].tolist())) < 2:
            continue
        v = Xb[tr][y[tr] == 1].mean(0) - Xb[tr][y[tr] == 0].mean(0)
        per_fam[held] = {
            "diffmeans_auroc": round(float(roc_auc_score(y[te], Xb[te] @ v)), 3),
            "length_auroc": round(float(roc_auc_score(y[te], plen[te])), 3),
            "mean_len_cert": round(float(plen[te][y[te] == 0].mean()), 1),
            "mean_len_unc": round(float(plen[te][y[te] == 1].mean()), 1),
        }
    report["per_held_family_at_bestlayer"] = {"layer": Lb, "families": per_fam}
    report["best_probe_layer"] = {"layer": best_probe, **per_layer[best_probe]}
    report["best_diffmeans_layer"] = {"layer": best_dom, **per_layer[best_dom]}

    # save the full-data diff-of-means direction (what BLADE/steering will use)
    dirs = {L: (acts[L].numpy()[y == 1].mean(0) - acts[L].numpy()[y == 0].mean(0)) for L in layers}
    dirs = {L: torch.tensor(v, dtype=torch.float32) for L, v in dirs.items()}
    DIRS.mkdir(parents=True, exist_ok=True)
    torch.save(dirs, DIRS / f"epistemic_{args.tag}.pt")
    (RESULTS / f"epistemic_direction_{args.tag}.json").write_text(json.dumps(report, indent=1))

    print(f"\nlength baseline AUROC (LOFO): {len_auc:.3f}")
    print("layer  probe_auroc  diffmeans_auroc")
    for L in layers:
        print(f"  {L:2d}      {per_layer[L]['probe_auroc']:.3f}        "
              f"{per_layer[L]['diffmeans_auroc']:.3f}")
    print(f"\nbest probe  L{best_probe}: {per_layer[best_probe]['probe_auroc']:.3f}")
    print(f"best diffmeans L{best_dom}: {per_layer[best_dom]['diffmeans_auroc']:.3f}")
    print(f"\nper-held-family @L{Lb}  (held-out family, direction trained on the rest):")
    print("  family        dom_auc  len_auc   len_cert/unc")
    for f, d in per_fam.items():
        print(f"  {f:12s}   {d['diffmeans_auroc']:.3f}    {d['length_auroc']:.3f}    "
              f"{d['mean_len_cert']}/{d['mean_len_unc']}")
    print(f"saved data/directions/epistemic_{args.tag}.pt + results/epistemic_direction_{args.tag}.json")


if __name__ == "__main__":
    main()
