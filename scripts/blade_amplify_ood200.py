"""BLADE amplify overlay for the steering-baselines figure, on the IDENTICAL 200 HarmBench
prompts / GEN_TOKENS / refusal judge / WikiText stream, run through the SAME locked selection
rule as the steering baselines (select alpha on in-dist val + XSTest + C4; freeze alpha*; report
the full factor grid on HarmBench). Supersedes the old 120-prompt curve.

BLADE mask is built on in-distribution data only (ELS L*=12, sparsity 0.005), same as
blade_refusal_amplify.py; HarmBench is queried only for reporting.
"""
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

from ttsafety.data import load_jsonl
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_nll
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments
from blade_refusal_amplify import scaled_weights
# reuse the exact same metrics/harness as the steering baselines
from blade_steering_baselines import (prefill_refusal, over_refusal, rel_ppl, wilson,
                                      L_BENIGN, BETA, PPL_TOKENS, C4_TOKENS, XSTEST_SEARCH)

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
L_STAR = [12]; SPARSITY = 0.005; COMPONENTS = "both"; N_DIR = 200
FACTORS = [1.3, 1.5, 2.0, 3.0]        # amplify grid (alpha>1); 1.0 is base, added separately
CONTRAST = [0.0]                       # BLADE removal, shown as a contrast point


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    harmful_val = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    wiki = load_wikitext_text(); c4 = load_c4_text()

    # ---- BLADE mask on in-dist data (identical recipe to blade_refusal_amplify) ----
    all_layers = list(range(len(get_decoder_layers(model))))
    directions = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    scores = score_edges(model, directions, muH, muU, L_STAR, COMPONENTS)
    sel = selection_from_ranking(rank_weight_indices(scores, 0.03), SPARSITY)
    n_edges = sum(len(v) for v in sel.values())
    print(f"BLADE mask {n_edges:,} weights (L*={L_STAR}, s={SPARSITY})", flush=True)

    def ctx(factor):
        @contextmanager
        def cm(mode):
            if factor == 1.0:
                yield
            else:
                with scaled_weights(model, sel, factor):
                    yield
        return cm

    @contextmanager
    def noop(mode):
        yield
    base_nll_wiki, n_scored = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_nll_c4, _ = teacher_forced_nll(model, tok, c4, max_tokens=C4_TOKENS)
    base_val = prefill_refusal(model, tok, harmful_val, noop)
    base_ood = prefill_refusal(model, tok, ood, noop)
    base_benign = over_refusal(model, tok, xstest[:XSTEST_SEARCH], noop)
    base_benign_full = over_refusal(model, tok, xstest, noop)
    print(f"base: val {base_val:.3f} OOD {base_ood:.3f} benign {base_benign_full:.3f}", flush=True)

    # ---- PHASE 1: select alpha* on validation (no HarmBench) ----
    search = []
    for f in FACTORS:
        c = ctx(f)
        r = prefill_refusal(model, tok, harmful_val, c)
        b = over_refusal(model, tok, xstest[:XSTEST_SEARCH], c)
        p = rel_ppl(model, tok, c4, base_nll_c4, c, C4_TOKENS)
        elig = (b <= base_benign + L_BENIGN) and (p <= BETA)
        search.append({"factor": f, "val_refusal": r, "benign": b, "c4_relppl": p, "eligible": elig})
        print(f"  val alpha={f:<4g} refusal {r:.3f} benign {b:.3f} c4Δppl {p:+.1%} {'OK' if elig else 'x'}", flush=True)
    elig = [row for row in search if row["eligible"]]
    a_star = (max(elig, key=lambda r: (r["val_refusal"], -r["c4_relppl"]))["factor"]
              if elig else 1.0)
    print(f"  -> frozen alpha*={a_star}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    frozen = {"alpha_star": a_star, "L_star": L_STAR, "sparsity": SPARSITY}
    sha = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    (RESULTS / "blade_amplify_ood200.frozen.json").write_text(
        json.dumps({"frozen": frozen, "sha256": sha}, indent=2))
    print(f"== FROZEN before HarmBench: {frozen} sha256 {sha} ==", flush=True)

    # ---- PHASE 2: HarmBench report over the full factor grid at the frozen mask ----
    rows = []
    for f in FACTORS + CONTRAST:
        c = ctx(f)
        n_ref = round(prefill_refusal(model, tok, ood, c) * len(ood))
        ood_ref = n_ref / len(ood)
        relppl = rel_ppl(model, tok, wiki, base_nll_wiki, c, PPL_TOKENS)
        benign_full = over_refusal(model, tok, xstest, c)
        report_elig = (benign_full <= base_benign_full + L_BENIGN) and (relppl <= BETA)
        lo, hi = wilson(n_ref, len(ood))
        rows.append({"factor": f, "ood_refusal": ood_ref, "ci": [lo, hi], "wiki_relppl": relppl,
                     "wiki_dnll": math.log(1 + relppl), "benign_full": benign_full,
                     "report_eligible_wiki_full_xstest": report_elig, "is_astar": (f == a_star)})
        print(f"  report alpha={f:<4g} OOD {ood_ref:.3f} [{lo:.2f},{hi:.2f}] Δppl {relppl:+.1%} "
              f"benign {benign_full:.3f} {'OK' if report_elig else 'x'}{'  <-a*' if f == a_star else ''}", flush=True)

    out = {"model": MODEL_ID, "n_edges": n_edges, "alpha_star": a_star,
           "frozen_config_sha256": sha,
           "base": {"val_refusal": base_val, "ood_refusal": base_ood, "benign": base_benign_full,
                    "wiki_nll": base_nll_wiki, "scored_tokens": n_scored},
           "val_search": search, "report": rows, "env": env_info()}
    (RESULTS / "blade_amplify_ood200.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved results/blade_amplify_ood200.json", flush=True)


if __name__ == "__main__":
    main()
