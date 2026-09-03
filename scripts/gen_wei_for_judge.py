"""Generate Wei-et-al.(2024) SNIP set-difference AMPLIFY outputs on XSTest and HarmBench, append to
the *_gens_for_judge.json files. Two configs: the val-frozen headline (q=p=0.001, alpha=3.0) and the
best report-benign-eligible point (alpha=2.0). Mask = matrixwise set difference of the precomputed
absolute-SNIP safety/utility scores; intervention = scale selected weights by alpha (same as BLADE
amplify). Mirrors gen_arditi_for_judge.py so the judge pipeline is identical."""
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.data import load_jsonl
from ttsafety.judge import is_over_refusal, is_refusal
from ttsafety.models import chat_wrap, load_model
from ttsafety.weight_prune import matrixwise_set_difference
from blade_refusal_amplify import scaled_weights
from blade_steering_baselines import _generate, GEN_TOKENS, PREFILL

DATA = Path("data"); RESULTS = Path("results")
SCORES = DATA / "weight_scores"
Q, P = 0.001, 0.001
CONFIGS = [("Wei_q001_p001_a3.0", 3.0), ("Wei_q001_p001_a2.0", 2.0)]


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    safety = torch.load(SCORES / "wei_safety_snip.pt", map_location="cpu", weights_only=False)["scores"]
    utility = torch.load(SCORES / "wei_utility_snip.pt", map_location="cpu", weights_only=False)["scores"]
    sel = matrixwise_set_difference(safety, utility, safety_fraction=Q, utility_fraction=P)
    print(f"Wei mask q={Q} p={P}: {sum(len(v) for v in sel.values())} edges", flush=True)

    @contextmanager
    def ctx(factor):
        with scaled_weights(model, sel, factor):
            yield

    for key, alpha in CONFIGS:
        with ctx(alpha):
            xo = _generate(model, tok, [chat_wrap(tok, p) for p in xstest], GEN_TOKENS)
        xsm = sum(is_over_refusal(o) for o in xo) / len(xo)
        xj = RESULTS / "xstest_gens_for_judge.json"; xd = json.loads(xj.read_text())
        xd[key] = {"stringmatch_overrefusal": xsm,
                   "items": [{"prompt": p, "output": o} for p, o in zip(xstest, xo)]}
        xj.write_text(json.dumps(xd, ensure_ascii=False, indent=1))

        with ctx(alpha):
            ho = _generate(model, tok, [chat_wrap(tok, p) + PREFILL for p in ood], GEN_TOKENS)
        conts = [PREFILL + o for o in ho]
        hsm = sum(is_refusal(c) for c in conts) / len(conts)
        hj = RESULTS / "harmbench_gens_for_judge.json"; hd = json.loads(hj.read_text())
        hd[key] = {"stringmatch_refusal": hsm,
                   "items": [{"request": p, "continuation": c} for p, c in zip(ood, conts)]}
        hj.write_text(json.dumps(hd, ensure_ascii=False, indent=1))
        print(f"{key} (a={alpha}): XSTest over-refusal {xsm:.3f} | HarmBench OOD refusal {hsm:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
