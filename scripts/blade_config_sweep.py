"""Does a different BLADE configuration avoid the amplify->over-refusal coupling?
Sweep L*, sparsity rho, and components (attn/mlp/both); for each, amplify (alpha>1) and measure
the OOD refusal (HarmBench 200) vs XSTest over-refusal (250) vs WikiText Δppl trade-off.
Question: is there any config with high OOD refusal at LOW XSTest over-refusal?"""
import json
from contextlib import contextmanager
from pathlib import Path

from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_nll
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import rank_weight_indices, selection_from_ranking
from blade_refusal_els import last_token_moments
from blade_refusal_amplify import scaled_weights
from blade_steering_baselines import prefill_refusal, over_refusal, rel_ppl, wilson, PPL_TOKENS

DATA = Path("data"); RESULTS = Path("results")
N_DIR = 200; SCREEN = 0.03
ALPHAS = [1.3, 1.5, 2.0]
# (name, L_star, rho, components)
CONFIGS = [
    ("L12 rho.001 both", [12], 0.001, "both"),
    ("L12 rho.002 both", [12], 0.002, "both"),
    ("L12 rho.005 both", [12], 0.005, "both"),   # the current default
    ("L12 rho.01  both", [12], 0.01, "both"),
    ("L12 rho.02  both", [12], 0.02, "both"),
    ("L15 rho.005 both", [15], 0.005, "both"),
    ("L22 rho.005 both", [22], 0.005, "both"),
    ("L12,15,22 rho.005 both", [12, 15, 22], 0.005, "both"),
    ("L12 rho.005 attn", [12], 0.005, "attn"),
    ("L12 rho.005 mlp",  [12], 0.005, "mlp"),
]


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    wiki = load_wikitext_text()

    all_layers = list(range(len(get_decoder_layers(model))))
    dirs = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:N_DIR], all_layers, "both", chat_wrap)
    muU = last_token_moments(model, tok, harmless[:N_DIR], all_layers, "both", chat_wrap)

    @contextmanager
    def noop(mode):
        yield
    base_nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_ood = prefill_refusal(model, tok, ood, noop)
    base_benign = over_refusal(model, tok, xstest, noop)
    print(f"base: OOD {base_ood:.3f} | XSTest over-refusal {base_benign:.3f}", flush=True)

    def ctx(sel, factor):
        @contextmanager
        def cm(mode):
            with scaled_weights(model, sel, factor):
                yield
        return cm

    results = {"base": {"ood": base_ood, "benign": base_benign}, "configs": []}
    for name, L, rho, comp in CONFIGS:
        scores = score_edges(model, dirs, muH, muU, L, comp)
        sel = selection_from_ranking(rank_weight_indices(scores, SCREEN), rho)
        n_edges = sum(len(v) for v in sel.values())
        rows = []
        for a in ALPHAS:
            c = ctx(sel, a)
            n_ref = round(prefill_refusal(model, tok, ood, c) * len(ood))
            ood_ref = n_ref / len(ood)
            benign = over_refusal(model, tok, xstest, c)
            relppl = rel_ppl(model, tok, wiki, base_nll, c, PPL_TOKENS)
            lo, hi = wilson(n_ref, len(ood))
            rows.append({"alpha": a, "ood": ood_ref, "ci": [lo, hi], "benign": benign,
                         "relppl": relppl})
            print(f"  {name:24} a={a:<4g} n={n_edges:>6} OOD {ood_ref:.3f} "
                  f"benign {benign:.3f} Δppl {relppl:+.1%}", flush=True)
        results["configs"].append({"name": name, "L": L, "rho": rho, "components": comp,
                                   "n_edges": n_edges, "sweep": rows})

    RESULTS.mkdir(exist_ok=True)
    results["env"] = env_info()
    (RESULTS / "blade_config_sweep.json").write_text(json.dumps(results, indent=2))
    print("saved results/blade_config_sweep.json", flush=True)


if __name__ == "__main__":
    main()
