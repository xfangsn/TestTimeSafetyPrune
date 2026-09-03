"""Amplify BLADE refusal weights (alpha>=1), with vs without activation steering
(re-extracted vec, fixed c), on OOD jailbreak refusal (HarmBench) + WikiText ppl."""
import json
from pathlib import Path
import torch
from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model, chat_wrap
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments
from blade_refusal_amplify import scaled_weights, prefill_refusal
from blade_amplify_refusal_ood import steer_add

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
L_STAR = [12]; STEER_L = 12; RHO = 0.005; N_DIR = 200; N_EVAL = 120
COMPONENTS = "both"; PPL_TOKENS = 5000
ALPHAS = [1.0, 1.3, 1.5, 2.0]; STEER_C = 0.2


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")][:N_EVAL]
    wiki = load_wikitext_text()

    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    sel = selection_from_ranking(rank_weight_indices(
        score_edges(model, dirs, muH, muU, L_STAR, COMPONENTS), max(RHO, 0.01)), RHO)
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    print(f"sel {sum(int(v.numel()) for v in sel.values()):,} edges | base ppl {base_ppl:.2f}", flush=True)

    rows = []
    for a in ALPHAS:
        with scaled_weights(model, sel, a):
            red = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])[STEER_L].float().cpu()
            r0 = prefill_refusal(model, tok, ood)
            p0 = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            with steer_add(model, STEER_L, red, STEER_C):
                r1 = prefill_refusal(model, tok, ood)
                p1 = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        rows.append({"alpha": a,
                     "amplify_refusal": r0, "amplify_ppl_delta": (p0 - base_ppl) / base_ppl,
                     "amplify_steer_refusal": r1, "amplify_steer_ppl_delta": (p1 - base_ppl) / base_ppl})
        print(f"  α={a:<4g} | amplify {r0:.3f} (Δppl {(p0-base_ppl)/base_ppl:+.1%}) | "
              f"+steer {r1:.3f} (Δppl {(p1-base_ppl)/base_ppl:+.1%})", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_amplify_steer_grid.json").write_text(json.dumps(
        {"model": MODEL_ID, "L_star": L_STAR, "steer_c": STEER_C, "base_ppl": base_ppl,
         "rows": rows}, indent=2))
    print("saved results/blade_amplify_steer_grid.json", flush=True)


if __name__ == "__main__":
    main()
