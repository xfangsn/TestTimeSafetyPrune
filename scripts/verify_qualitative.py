"""Verify the over-refusal finding is real, not a judge artifact:
(1) dump actual generations under BLADE amplify on SAFE XSTest prompts;
(2) re-measure over-refusal on a SECOND, held-out benign set (harmless[256:320], unused in
    extraction) for base / BLADE α=1.3,1.5 / CAA c* / ActAdd c*."""
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
from blade_steering_baselines import build_methods, over_refusal, _generate, GEN_TOKENS

DATA = Path("data")


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    harmless_held = harmless[256:320]                       # 64, unused in extraction
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]

    methods, _ = build_methods(model, tok)
    # BLADE mask (same recipe as blade_amplify_ood200)
    all_layers = list(range(len(get_decoder_layers(model))))
    dirs = extract_refusal_direction(model, tok, harmful[:200], harmless[:192])
    muH = last_token_moments(model, tok, harmful[:200], all_layers, "both", chat_wrap)
    muU = last_token_moments(model, tok, harmless[:200], all_layers, "both", chat_wrap)
    sel = selection_from_ranking(rank_weight_indices(
        score_edges(model, dirs, muH, muU, [12], "both"), 0.03), 0.005)

    @contextmanager
    def blade(factor):
        with scaled_weights(model, sel, factor):
            yield

    @contextmanager
    def noop():
        yield

    # (2) second held-out benign set
    print("\n=== over-refusal on a 2nd held-out benign set (harmless[256:320], n=64) vs XSTest ===")
    def held(cm):  # cm is a no-arg context manager
        with cm:
            outs = _generate(model, tok, [chat_wrap(tok, p) for p in harmless_held], GEN_TOKENS)
        return sum(is_over_refusal(o) for o in outs) / len(outs)
    configs = [("base", noop()), ("BLADE a=1.3", blade(1.3)), ("BLADE a=1.5", blade(1.5)),
               ("CAA c*=0.5 L20", methods["caa"]["ctx"](20, 0.5)("plain_gen")),
               ("ActAdd c*=8 L14", methods["actadd"]["ctx"](14, 8.0)("plain_gen"))]
    for name, cm in configs:
        print(f"  {name:18} held-out harmless over-refusal {held(cm):.3f}")

    # (1) qualitative dump under BLADE a=1.5 vs base on 12 safe prompts
    print("\n=== sample generations on SAFE XSTest prompts (BLADE a=1.5) ===")
    sample = xstest[:12]
    with blade(1.5):
        amp = _generate(model, tok, [chat_wrap(tok, p) for p in sample], GEN_TOKENS)
    base = _generate(model, tok, [chat_wrap(tok, p) for p in sample], GEN_TOKENS)
    for p, a, b in zip(sample, amp, base):
        print(f"\n  PROMPT: {p}")
        print(f"    base : {b[:110].strip()!r}")
        print(f"    a=1.5: {a[:110].strip()!r}  [over-refusal={is_over_refusal(a)}]")


if __name__ == "__main__":
    main()
