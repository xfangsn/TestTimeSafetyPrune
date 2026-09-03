"""Re-generate the 9/11-hoax example at a larger token budget (the eval used only 64 tokens),
for a fuller appendix exhibit. Saves results/example_911_long.json."""
import json
from contextlib import contextmanager
from pathlib import Path

from ttsafety.data import load_jsonl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments
from blade_refusal_amplify import scaled_weights
from blade_steering_baselines import build_methods, _generate, PREFILL

DATA = Path("data"); RESULTS = Path("results")
MAX_NEW = 300


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    prompt = ood[68]  # index 69 (1-based)
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    methods, _ = build_methods(model, tok)
    all_layers = list(range(len(get_decoder_layers(model))))
    dirs = extract_refusal_direction(model, tok, harmful[:200], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:200], all_layers, "both", chat_wrap)
    muU = last_token_moments(model, tok, harmless[:200], all_layers, "both", chat_wrap)
    sel = selection_from_ranking(rank_weight_indices(
        score_edges(model, dirs, muH, muU, [14], "both"), 0.03), 0.002)

    @contextmanager
    def noop():
        yield
    wrapped = [chat_wrap(tok, prompt) + PREFILL]
    configs = {
        "base": noop(),
        "ActAdd_c8": methods["actadd"]["ctx"](14, 8.0)("prefill_attack"),
        "CAA_c0.5": methods["caa"]["ctx"](20, 0.5)("prefill_attack"),
        "RefDir_c0.35": methods["arditi"]["ctx"](16, 0.35)("prefill_attack"),
        "BLADE_r002": scaled_weights(model, sel, 1.5),
    }
    out = {"prompt": prompt, "max_new": MAX_NEW, "responses": {}}
    for name, cm in configs.items():
        with cm:
            g = _generate(model, tok, wrapped, MAX_NEW)[0]
        out["responses"][name] = PREFILL + g
        print(f"\n===== {name} ({len(g)} chars) =====\n{PREFILL + g}", flush=True)
    (RESULTS / "example_911_long.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nsaved results/example_911_long.json", flush=True)


if __name__ == "__main__":
    main()
