"""Generate XSTest safe-prompt responses under the key configs, for LLM-judge re-scoring of
over-refusal (cross-check of the string-match is_over_refusal). Saves prompt+output per config."""
import json
from contextlib import contextmanager
from pathlib import Path

from ttsafety.data import load_jsonl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.judge import is_over_refusal
from ttsafety.models import chat_wrap, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments
from blade_refusal_amplify import scaled_weights
from blade_steering_baselines import build_methods, _generate, GEN_TOKENS

DATA = Path("data"); RESULTS = Path("results")


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]

    methods, _ = build_methods(model, tok)
    all_layers = list(range(len(get_decoder_layers(model))))
    dirs = extract_refusal_direction(model, tok, harmful[:200], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:200], all_layers, "both", chat_wrap)
    muU = last_token_moments(model, tok, harmless[:200], all_layers, "both", chat_wrap)

    def blade_sel(L, rho):
        return selection_from_ranking(rank_weight_indices(
            score_edges(model, dirs, muH, muU, L, "both"), 0.03), rho)
    sel_L14 = blade_sel([14], 0.002)
    sel_L12 = blade_sel([12], 0.005)

    @contextmanager
    def noop():
        yield

    configs = {
        "base": noop(),
        "BLADE_L14_r002_a1.5": scaled_weights(model, sel_L14, 1.5),
        "BLADE_L12_r005_a1.5": scaled_weights(model, sel_L12, 1.5),
        "CAA_L20_c0.5": methods["caa"]["ctx"](20, 0.5)("plain_gen"),
        "ActAdd_L14_c8": methods["actadd"]["ctx"](14, 8.0)("plain_gen"),
    }
    wrapped = [chat_wrap(tok, p) for p in xstest]
    out = {}
    for name, cm in configs.items():
        with cm:
            outs = _generate(model, tok, wrapped, GEN_TOKENS)
        sm = sum(is_over_refusal(o) for o in outs) / len(outs)
        out[name] = {"stringmatch_overrefusal": sm,
                     "items": [{"prompt": p, "output": o} for p, o in zip(xstest, outs)]}
        print(f"{name:22} string-match over-refusal {sm:.3f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "xstest_gens_for_judge.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved results/xstest_gens_for_judge.json", flush=True)


if __name__ == "__main__":
    main()
