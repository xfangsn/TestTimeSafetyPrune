"""Cross-model BLADE-G + ELS on the epistemic-uncertainty (hedging) behavior — v2 protocol.

Same pipeline as scripts/blade_epistemic_els.py (last-prompt-token contrast, Arditi direction,
last-token writer moments, solo-pool -> best-first ELS, BLADE-G g1scalar scoring, REMOVE sweep
over rho in {0.0005, 0.002, 0.005, 0.02}) but on the **v2 protocol**:

  * data  = results/epistemic_pairs_v2.json, split entity-disjointly THREE ways with
    blade_epistemic_p0.split_3way(rows) -> (train, select, test);
  * train  (uncertain/certain) -> refusal-style direction + last-token writer moments;
  * select (uncertain only)    -> the ELS metric (solo-pool screen + best-first), so layer
    selection never touches the reported set;
  * test   (uncertain/certain) -> the REPORTED numbers: base_hedge_unanswerable /
    base_hedge_known and every remove_sweep row's hedge_unanswerable / hedge_known.

Removal target: zeroing top-BLADE weights should REDUCE hedging on unanswerable prompts
(the model answers confidently instead), while hedging on known-answer prompts and the
C4/WikiText perplexity budget are tracked as side effects.

This is the run that feeds the **Uncertainty row of the cross-model S3 table**
(scripts/plot_s3_crossmodel_wide.py -> scripts/make_s3_table.py), which reads the
rho=0.005 entry of remove_sweep from results/blade_epistemic_els_<tag>_bladeg_v2.json.

Wrap: Qwen models use the thinking-disabled chat template; other instruct models use
ttsafety.models.chat_wrap; base/-pt models use a plain-prompt wrap (as blade_refusal_els.py).

Env:
  BLADE_MODEL      model id (default Qwen/Qwen3-8B)
  L_STAR           comma-separated layer ints; pins L* and SKIPS ELS (solo pool + best-first)
  BLADE_AMPLIFY=1  also run the (slow) amplify sweep; off by default
  BLADE_SCREEN_FRAC / BLADE_TESTFRAC / BLADE_BETA / BLADE_EPS / BLADE_AMP_ALPHAS

Usage:
  BLADE_MODEL=meta-llama/Llama-3.2-3B-Instruct L_STAR=14,1 .venv/bin/python scripts/blade_epistemic_els_v2.py
"""
import json
import os
from pathlib import Path

import torch

import ttsafety.extract as EX
import ttsafety.generate as GEN
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.generate import generate_texts
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges, score_edges_g
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_prune import (pruned_weights, rank_weight_indices,
                                   selection_from_ranking)
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import (UNC_MARKERS, is_unc, last_token_moments, qwen_wrap)
from blade_epistemic_p0 import split_3way

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BLADE_G = True                                    # primary method: score_edges_g (g1scalar)
Q_GLOBAL = None                                   # generic-importance Q on all layers (set in main)
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
COMPONENTS = "both"
SCREEN_FRAC = float(os.environ.get("BLADE_SCREEN_FRAC", "0.005"))
GREEDY_TESTFRAC = float(os.environ.get("BLADE_TESTFRAC", "0.005"))
BETA = float(os.environ.get("BLADE_BETA", "0.05"))
EPS = float(os.environ.get("BLADE_EPS", "0.005"))
PPL_TOKENS = 5_000
GEN_TOKENS = 64
AMP_ALPHAS = [float(x) for x in os.environ.get("BLADE_AMP_ALPHAS", "1.25,1.5").split(",")]
DO_AMPLIFY = os.environ.get("BLADE_AMPLIFY", "") == "1"
REMOVE_FRACS = [0.0005, 0.002, 0.005, 0.02]

assert UNC_MARKERS, "hedge marker list must be non-empty"


def _l_star_env():
    """L_STAR=22,9,28 -> [22, 9, 28]; empty/unset -> None."""
    s = os.environ.get("L_STAR", "").strip()
    if not s:
        return None
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def unc_rate(model, tok, prompts, max_new=GEN_TOKENS, bs=16):
    outs = generate_texts(model, tok, list(prompts), max_new_tokens=max_new, batch_size=bs)
    return sum(is_unc(o) for o in outs) / max(len(outs), 1)


def _med_pos(d):
    t = torch.cat([v.flatten() for v in d.values()]).float(); return t[t > 0].median().item()


def _score_fn_for(model, directions, mu_a, mu_b, all_layers, components):
    if not BLADE_G:
        return score_edges
    lam = _med_pos(score_edges(model, directions, mu_a, mu_b, all_layers, components)) / _med_pos(Q_GLOBAL)

    def sfn(m, d, a, b, layers, comp):
        S = score_edges_g(m, d, a, b, layers, comp, Q=Q_GLOBAL, lam=lam, abstain=True)
        return {k: torch.where(torch.isfinite(v), v, torch.zeros_like(v)) for k, v in S.items()}
    return sfn


def main():
    mid = MODEL_ID.lower()
    is_qwen = "qwen" in mid
    is_base = mid.endswith("-pt") or "-base" in mid
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    wrap = chat_wrap
    if is_qwen:                       # disable thinking for direction/moments/gen
        wrap = qwen_wrap
        EX.chat_wrap = qwen_wrap
        GEN.chat_wrap = qwen_wrap
    if is_base:                       # pretrained: no chat template, plain prompt
        base_wrap = lambda tokenizer, s: s + "\n"
        wrap = base_wrap
        EX.chat_wrap = base_wrap
        GEN.chat_wrap = base_wrap
        print("[base mode] plain prompt wrap", flush=True)
    n_layers = len(get_decoder_layers(model))
    all_layers = list(range(n_layers))

    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, sel_rows, test = split_3way(rows)
    unc_tr = [r["question"] for r in tr if r["label"] == 1]
    cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_sel = [r["question"] for r in sel_rows if r["label"] == 1]      # ELS metric set
    unc_test = [r["question"] for r in test if r["label"] == 1]         # reported
    cert_test = [r["question"] for r in test if r["label"] == 0]        # reported

    c4 = load_c4_text(); wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_unc_uncset = unc_rate(model, tok, unc_test)    # hedging on UNANSWERABLE (removal target: down)
    base_unc_certset = unc_rate(model, tok, cert_test)  # hedging on KNOWN (side effect / amplify target)
    print(f"{MODEL_ID}: {n_layers}L | ppl {base_ppl:.2f} | base hedge (test): unanswerable "
          f"{base_unc_uncset:.3f}  known {base_unc_certset:.3f}", flush=True)
    print(f"  split: dir_train {len(tr)} ({len(unc_tr)}unc/{len(cert_tr)}cert) | "
          f"els_select {len(unc_sel)}unc | test {len(unc_test)}unc/{len(cert_test)}cert", flush=True)

    report = {"model": MODEL_ID, "n_layers": n_layers, "blade_g": BLADE_G,
              "data": "epistemic_pairs_v2.json", "split": "split_3way",
              "base_ppl_c4": base_ppl, "base_ppl_wiki": base_ppl_wiki, "beta": BETA,
              "base_hedge_unanswerable": base_unc_uncset, "base_hedge_known": base_unc_certset,
              "n_train": [len(unc_tr), len(cert_tr)], "n_sel": len(unc_sel),
              "n_eval": [len(unc_test), len(cert_test)],
              "env": env_info()}

    print("direction + last-token moments (uncertain vs certain, train split) ...", flush=True)
    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, wrap)

    global Q_GLOBAL
    print("generic-importance Q (g1scalar) ...", flush=True)
    Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4,
                                                seqlen=2048, batch_size=2, mode="g1scalar",
                                                max_tokens=65536)
    score_fn = _score_fn_for(model, directions, muUNC, muCERT, all_layers, COMPONENTS)

    def ppl_now():
        return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    pinned = _l_star_env()
    if pinned is not None:
        L_star = pinned
        report["candidate_pool"] = None
        report["L_star_source"] = "pinned"
        print(f"PINNED L*={L_star} (skipping solo pool + best-first)", flush=True)
    else:
        base_sel = unc_rate(model, tok, unc_sel)      # ELS base metric on the SELECT split

        def measure():                               # ELS metric = hedge rate on select split
            return unc_rate(model, tok, unc_sel), ppl_now()

        print(f"ELS: solo pool -> best-first (metric=hedge on select, base {base_sel:.3f}) ...",
              flush=True)
        pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS,
                               ppl_now, base_ppl, screen_frac=SCREEN_FRAC, beta=BETA,
                               score_fn=score_fn)
        L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS,
                                  measure, base_sel, base_ppl, beta=BETA, eps=EPS,
                                  test_frac=GREEDY_TESTFRAC, score_fn=score_fn)
        print(f"pool={len(pool)} -> L* = {L_star if L_star else 'NONE'}", flush=True)
        report["candidate_pool"] = pool
        report["L_star_source"] = "els"
        report["base_hedge_select"] = base_sel
    report["L_star"] = L_star

    if L_star:
        scores = score_fn(model, directions, muUNC, muCERT, L_star, COMPONENTS)
        rk = rank_weight_indices(scores, 0.03)
        sweep = []
        for frac in REMOVE_FRACS:
            sel = selection_from_ranking(rk, frac)
            n = sum(int(v.numel()) for v in sel.values())
            with pruned_weights(model, sel):                     # REMOVE (alpha=0)
                hu = unc_rate(model, tok, unc_test)              # hedging on unanswerable (want DOWN)
                hk = unc_rate(model, tok, cert_test)             # hedging on known (side effect)
                ppl_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
                ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            row = {"sparsity": frac, "n_edges": n, "hedge_unanswerable": hu, "hedge_known": hk,
                   "ppl_delta": (ppl_c4 - base_ppl) / base_ppl,
                   "ppl_delta_wiki": (ppl_wiki - base_ppl_wiki) / base_ppl_wiki}
            sweep.append(row)
            print(f"  REMOVE s={frac:.2%} hedge unans {base_unc_uncset:.2f}->{hu:.2f} "
                  f"known {base_unc_certset:.2f}->{hk:.2f}  ΔpplC4 {row['ppl_delta']:+.2%} "
                  f"ΔpplWiki {row['ppl_delta_wiki']:+.2%}", flush=True)
        report["remove_sweep"] = sweep

        # amplify (raw alphaW baseline). Opt-in: BLADE_AMPLIFY=1. Target: inject hedging on KNOWN.
        if DO_AMPLIFY:
            amp = []
            for a in AMP_ALPHAS:
                for frac in [0.002, 0.005]:
                    sel = selection_from_ranking(rk, frac)
                    with scaled_weights(model, sel, a):
                        hk = unc_rate(model, tok, cert_test)
                        hu = unc_rate(model, tok, unc_test)
                        ppl_c4 = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
                    amp.append({"alpha": a, "sparsity": frac, "hedge_known": hk,
                                "hedge_unanswerable": hu,
                                "ppl_delta": (ppl_c4 - base_ppl) / base_ppl})
                    print(f"  AMPLIFY a={a} s={frac:.2%} hedge known {base_unc_certset:.2f}->{hk:.2f} "
                          f"unans {hu:.2f}  ΔpplC4 {(ppl_c4-base_ppl)/base_ppl:+.2%}", flush=True)
            report["amplify_sweep"] = amp

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    out = RESULTS / f"blade_epistemic_els_{tag}_bladeg_v2.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
