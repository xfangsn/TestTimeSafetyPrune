"""Amplify BLADE-selected refusal weights (alpha>1) and test OOD refusal (HarmBench),
plus the RE-EXTRACTED-vector analysis (symmetric to the removal steering result):
- removal (alpha=0) erases the refusal direction  -> re-extracting it fails/degenerates;
- amplification (alpha>1) preserves/strengthens it -> re-extracting it stays valid, and
  steering with the re-extracted vector on the amplified model still raises refusal.
Refusal at ceiling on plain prompts, so we use a jailbreak/prefill refusal metric (headroom).
"""
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments, refusal_rate
from blade_refusal_amplify import scaled_weights, prefill_refusal

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
COMPONENTS = "both"; PPL_TOKENS = 5000; RHO = 0.005; N_DIR = 200; N_EVAL = 120
L_STAR = [12]; STEER_L = 12
FACTORS = [1.0, 1.3, 1.5, 2.0, 3.0, 0.0]   # 1.0 base, >1 amplify, 0.0 removal (contrast)
STEER_C = [0.0, 1.0, 2.0, 4.0]


@contextmanager
def steer_add(model, layer_idx, vec, coef):   # ADD +coef*vec (strengthen refusal)
    if coef == 0.0:
        yield; return
    v = vec.to(model.device)
    def hook(_m, _i, out):
        if isinstance(out, tuple):
            out[0].add_(coef * v.to(out[0].dtype)); return out
        return out + coef * v.to(out.dtype)
    h = get_decoder_layers(model)[layer_idx].register_forward_hook(hook)
    try: yield
    finally: h.remove()


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    hv = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    in_test = [r["instruction"] for r in load_jsonl(DATA / "harmful_test.jsonl")][:N_EVAL]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")][:N_EVAL]
    wiki = load_wikitext_text()

    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])
    from ttsafety.models import chat_wrap
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    sel = selection_from_ranking(rank_weight_indices(
        score_edges(model, dirs, muH, muU, L_STAR, COMPONENTS), max(RHO, 0.01)), RHO)
    n_edges = sum(int(v.numel()) for v in sel.values())
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_dir = dirs[STEER_L].float()
    print(f"L*={L_STAR} sel {n_edges:,} edges | base ppl {base_ppl:.2f}", flush=True)

    # --- amplify sweep: OOD (HarmBench) + in-dist prefill/plain refusal + ppl + re-extract cosine ---
    rows = []
    for f in FACTORS:
        with scaled_weights(model, sel, f):
            ood_pref = prefill_refusal(model, tok, ood)
            ood_plain = refusal_rate(model, tok, ood)
            ind_pref = prefill_refusal(model, tok, in_test)
            ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            red = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])[STEER_L].float()
        cos = F.cosine_similarity(red.cpu(), base_dir.cpu(), dim=0).item()
        dppl = (ppl - base_ppl) / base_ppl
        rows.append({"factor": f, "ood_prefill_refusal": ood_pref, "ood_plain_refusal": ood_plain,
                     "indist_prefill_refusal": ind_pref, "ppl_delta": dppl,
                     "reextract_cos_base": cos})
        tag = "REMOVAL" if f == 0 else ("base" if f == 1 else f"amplify x{f:g}")
        print(f"  {tag:12} f={f:<4g} OOD prefill {ood_pref:.3f} plain {ood_plain:.3f} | "
              f"in-dist prefill {ind_pref:.3f} | Δppl {dppl:+.1%} | cos(re-extract,base)={cos:+.3f}",
              flush=True)

    # --- steering on the AMPLIFIED model (f=1.3) with the RE-EXTRACTED vector, OOD ---
    steer = []
    with scaled_weights(model, sel, 1.3):
        red = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])[STEER_L].float().cpu()
        for c in STEER_C:
            with steer_add(model, STEER_L, red, c):
                op = prefill_refusal(model, tok, ood)
            steer.append({"coef": c, "ood_prefill_refusal": op})
            print(f"  [amplify1.3 + steer(re-extract) c={c:<3g}] OOD prefill {op:.3f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_amplify_refusal_ood.json").write_text(json.dumps(
        {"model": MODEL_ID, "L_star": L_STAR, "steer_layer": STEER_L, "rho": RHO,
         "n_edges": n_edges, "base_ppl": base_ppl, "amplify": rows,
         "amplify1.3_reextract_steer": steer, "env": env_info()}, indent=2))
    print("saved results/blade_amplify_refusal_ood.json", flush=True)


if __name__ == "__main__":
    main()
