"""Minimal joint-amplification test: refusal and uncertainty masks, localized SEPARATELY on the SAME
model, amplified alone and together.

Why Llama-3.2-3B: its two selected layer sets are DISJOINT (refusal L*=[13], uncertainty L*=[14,1]).
BLADE only scores weights inside L* (score_edges_g iterates iter_residual_writers(model, L*)), so the two
masks share ZERO coordinates *by construction*, not merely "few". That isolates the question: with
double-scaling of a shared weight structurally impossible, is joint amplification clean? If it is NOT,
weight overlap is not the variable that governs composition.

Masks are FROZEN (selected once at the removal-semantics lambda, rho fixed) and only alpha is swept, so
the mask -- and therefore the overlap -- is identical across cells. Amplify-optimal re-selection
(lambda_eff = lambda*(alpha-1)) is deliberately NOT done here; this is the frozen-mask dose sweep.

Axes (benefit AND harm for each behavior, since both behaviors are abstention-like and "both improved"
is trivially achievable by declining more):
  refusal   benefit = OOD refusal on HarmBench      harm = XSTest-safe over-refusal
  uncertain benefit = hedge on held-out unanswerable harm = hedge on held-out known
  capability = held-out WikiText delta-ppl + lexical degeneration rate
Keyword/marker judges are cheap proxies; a judged pass is a follow-up, not this script.

Env: BLADE_MODEL, RHO, ALPHAS, N_OOD, N_XSTEST, OUT_TAG.
Usage: PYTHONPATH=src:scripts .venv/bin/python scripts/joint_amplify_minimal.py
"""
import json, os
from contextlib import ExitStack
from pathlib import Path

import torch

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.generate import generate_texts
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.hooks import get_decoder_layers
from ttsafety.judge import is_over_refusal, is_refusal
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import is_unc, last_token_moments, qwen_wrap
from blade_epistemic_p0 import split_3way

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
COMPONENTS = "both"
RHO = float(os.environ.get("RHO", "0.005"))
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", "1.5,2.0").split(",")]
PPL_TOKENS = 5_000
N_DIR = 200
N_OOD = int(os.environ.get("N_OOD", "100"))
N_XSTEST = int(os.environ.get("N_XSTEST", "100"))
GEN_TOK = 64


def rep_score(t):
    w = t.split()
    if len(w) < 8:
        return 1.0 if len(w) < 3 else 0.0
    g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g)


def _med_pos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float()
    return t[t > 0].median().item()


def build_mask(model, directions, mu_a, mu_b, L_star, Q, all_layers, rho):
    """BLADE-G mask on L_star at sparsity rho (removal-semantics lambda, fixed across the alpha sweep)."""
    lam = _med_pos(score_edges(model, directions, mu_a, mu_b, all_layers, COMPONENTS)) / _med_pos(Q)
    S = score_edges_g(model, directions, mu_a, mu_b, L_star, COMPONENTS, Q=Q, lam=lam, abstain=True)
    S = {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
    sel = selection_from_ranking(rank_weight_indices(S, max(rho, 0.01)), rho)
    return sel, lam


def overlap(sel_a, sel_b):
    """Shared (writer, flat-index) coordinates between two masks."""
    n = 0
    shared_writers = sorted(set(sel_a) & set(sel_b))
    for name in shared_writers:
        a = set(sel_a[name].tolist()); b = set(sel_b[name].tolist())
        n += len(a & b)
    return {"shared_writers": shared_writers, "n_shared_coords": n,
            "n_ref": sum(int(v.numel()) for v in sel_a.values()),
            "n_unc": sum(int(v.numel()) for v in sel_b.values())}


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    c4, wiki = load_c4_text(), load_wikitext_text()
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)

    print("generic-importance Q (g1scalar) ...", flush=True)
    Q, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4, seqlen=2048,
                                         batch_size=2, mode="g1scalar", max_tokens=65536)

    # ---- refusal side (native chat wrap, harmful vs harmless at last prompt token) ----
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    L_ref = json.loads((RESULTS / f"blade_refusal_els_{tag}_bladeg.json").read_text())["L_star"]
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    EX.chat_wrap = GEN.chat_wrap = chat_wrap
    dir_ref = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    sel_ref, lam_ref = build_mask(model, dir_ref, muH, muU, L_ref, Q, all_layers, RHO)

    # ---- uncertainty side (qwen_wrap protocol, uncertain vs certain, entity-disjoint 3-way split) ----
    L_unc = json.loads((RESULTS / f"blade_epistemic_els_{tag}_bladeg_v2.json").read_text())["L_star"]
    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, _sel_rows, test = split_3way(rows)
    unc_tr = [r["question"] for r in tr if r["label"] == 1]
    cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_test = [r["question"] for r in test if r["label"] == 1]
    cert_test = [r["question"] for r in test if r["label"] == 0]
    EX.chat_wrap = GEN.chat_wrap = qwen_wrap
    dir_unc = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    sel_unc, lam_unc = build_mask(model, dir_unc, muUNC, muCERT, L_unc, Q, all_layers, RHO)

    ov = overlap(sel_ref, sel_unc)
    print(f"L*_ref={L_ref}  L*_unc={L_unc}", flush=True)
    print(f"OVERLAP: {ov['n_shared_coords']} shared coords "
          f"(|S_ref|={ov['n_ref']}, |S_unc|={ov['n_unc']}, shared writers={ov['shared_writers']})", flush=True)

    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")][:N_OOD]
    xsafe = [r["prompt"] if "prompt" in r else r["instruction"]
             for r in load_jsonl(DATA / "xstest_safe.jsonl")][:N_XSTEST]

    def measure(label):
        """All four axes + capability, under whatever edit context is active."""
        EX.chat_wrap = GEN.chat_wrap = chat_wrap
        g_ood = generate_texts(model, tok, ood, max_new_tokens=GEN_TOK, batch_size=16)
        g_xs = generate_texts(model, tok, xsafe, max_new_tokens=GEN_TOK, batch_size=16)
        EX.chat_wrap = GEN.chat_wrap = qwen_wrap
        g_un = generate_texts(model, tok, unc_test, max_new_tokens=GEN_TOK, batch_size=16)
        g_kn = generate_texts(model, tok, cert_test, max_new_tokens=GEN_TOK, batch_size=16)
        alltxt = g_ood + g_xs + g_un + g_kn
        ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        rec = {"cond": label,
               "ood_refusal": sum(map(is_refusal, g_ood)) / len(g_ood),
               "xstest_overrefusal": sum(map(is_over_refusal, g_xs)) / len(g_xs),
               "hedge_unanswerable": sum(map(is_unc, g_un)) / len(g_un),
               "hedge_known": sum(map(is_unc, g_kn)) / len(g_kn),
               "ppl_delta_wiki": (ppl - base_ppl_wiki) / base_ppl_wiki,
               "degen": sum(rep_score(t) > 0.5 for t in alltxt) / len(alltxt)}
        print(f"  {label:22} OODref {rec['ood_refusal']:.3f} XSover {rec['xstest_overrefusal']:.3f} "
              f"hedgeUN {rec['hedge_unanswerable']:.3f} hedgeKN {rec['hedge_known']:.3f} "
              f"dppl {rec['ppl_delta_wiki']:+.2%} degen {rec['degen']:.2f}", flush=True)
        return rec

    cells = [measure("base")]
    for a in ALPHAS:
        with scaled_weights(model, sel_ref, a):
            cells.append(measure(f"ref_only_a{a}"))
        with scaled_weights(model, sel_unc, a):
            cells.append(measure(f"unc_only_a{a}"))
        # disjoint masks -> nesting never double-scales a coordinate (asserted above)
        with ExitStack() as st:
            st.enter_context(scaled_weights(model, sel_ref, a))
            st.enter_context(scaled_weights(model, sel_unc, a))
            cells.append(measure(f"both_a{a}"))

    out = {"model": MODEL_ID, "rho": RHO, "alphas": ALPHAS, "L_star_refusal": L_ref,
           "L_star_uncertainty": L_unc, "overlap": ov, "lam_refusal": lam_ref,
           "lam_uncertainty": lam_unc, "masks": "frozen (removal-semantics lambda); alpha swept",
           "n_ood": len(ood), "n_xstest": len(xsafe), "n_unc_test": len(unc_test),
           "n_known_test": len(cert_test), "base_ppl_wiki": base_ppl_wiki,
           "cells": cells, "env": env_info()}
    p = RESULTS / f"joint_amplify_minimal{os.environ.get('OUT_TAG','')}_{tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
