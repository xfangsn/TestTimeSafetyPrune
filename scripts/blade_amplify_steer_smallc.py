"""Follow-up: on the AMPLIFIED (x1.3) refusal model, steer with the RE-EXTRACTED
refusal vector at SMALL coefficients (raw mean-diff is large; c>=1 over-steered).
Shows the re-extracted vector on the amplified model is usable to further raise OOD
jailbreak refusal."""
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
SMALL_C = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")][:N_EVAL]

    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, COMPONENTS, chat_wrap)
    sel = selection_from_ranking(rank_weight_indices(
        score_edges(model, dirs, muH, muU, L_STAR, COMPONENTS), max(RHO, 0.01)), RHO)

    rows = []
    with scaled_weights(model, sel, 1.3):
        red = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])[STEER_L].float().cpu()
        for c in SMALL_C:
            with steer_add(model, STEER_L, red, c):
                op = prefill_refusal(model, tok, ood)
            rows.append({"coef": c, "ood_prefill_refusal": op})
            print(f"  amplify1.3 + steer(re-extract) c={c:<4g}  OOD prefill {op:.3f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_amplify_steer_smallc.json").write_text(json.dumps(
        {"model": MODEL_ID, "L_star": L_STAR, "steer_layer": STEER_L, "rows": rows}, indent=2))
    print("saved results/blade_amplify_steer_smallc.json", flush=True)


if __name__ == "__main__":
    main()
