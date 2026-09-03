"""Generate HarmBench prefill-attack continuations under the same 5 configs as the XSTest
judge-agreement figure, for LLM-judge cross-check of OOD refusal rate (string-match is_refusal
vs LLM judge). Saves the harmful request + full continuation per item."""
import json
from contextlib import contextmanager
from pathlib import Path

from ttsafety.data import load_jsonl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments
from blade_refusal_amplify import scaled_weights
from blade_steering_baselines import build_methods, _generate, GEN_TOKENS, PREFILL

DATA = Path("data"); RESULTS = Path("results")


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]

    methods, _ = build_methods(model, tok)
    all_layers = list(range(len(get_decoder_layers(model))))
    dirs = extract_refusal_direction(model, tok, harmful[:200], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:200], all_layers, "both", chat_wrap)
    muU = last_token_moments(model, tok, harmless[:200], all_layers, "both", chat_wrap)

    def blade_sel(L, rho):
        return selection_from_ranking(rank_weight_indices(
            score_edges(model, dirs, muH, muU, L, "both"), 0.03), rho)
    sel_L14 = blade_sel([14], 0.002); sel_L12 = blade_sel([12], 0.005)

    @contextmanager
    def noop():
        yield

    # each value is a (context-manager, needs-mode-flag) — steering ctx needs "prefill_attack"
    configs = {
        "base": ("cm", noop()),
        "BLADE_L14_r002_a1.5": ("cm", scaled_weights(model, sel_L14, 1.5)),
        "BLADE_L12_r005_a1.5": ("cm", scaled_weights(model, sel_L12, 1.5)),
        "CAA_L20_c0.5": ("cm", methods["caa"]["ctx"](20, 0.5)("prefill_attack")),
        "ActAdd_L14_c8": ("cm", methods["actadd"]["ctx"](14, 8.0)("prefill_attack")),
    }
    wrapped = [chat_wrap(tok, p) + PREFILL for p in ood]
    out = {}
    for name, (_, cm) in configs.items():
        with cm:
            outs = _generate(model, tok, wrapped, GEN_TOKENS)
        conts = [PREFILL + o for o in outs]
        sm = sum(is_refusal(c) for c in conts) / len(conts)
        out[name] = {"stringmatch_refusal": sm,
                     "items": [{"request": p, "continuation": c} for p, c in zip(ood, conts)]}
        print(f"{name:22} string-match OOD refusal {sm:.3f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "harmbench_gens_for_judge.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved results/harmbench_gens_for_judge.json", flush=True)


if __name__ == "__main__":
    main()
