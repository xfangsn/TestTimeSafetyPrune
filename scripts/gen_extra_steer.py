"""Generate the next-coefficient points for CAA (L20,c=0.7) and ActAdd (L14,c=16) to show these
methods plateau/collapse and cannot reach BLADE's OOD refusal. Append to *_gens_for_judge.json."""
import json
from pathlib import Path

from ttsafety.data import load_jsonl
from ttsafety.judge import is_over_refusal, is_refusal
from ttsafety.models import chat_wrap, load_model
from blade_steering_baselines import build_methods, _generate, GEN_TOKENS, PREFILL

DATA = Path("data"); RESULTS = Path("results")
CONFIGS = [("CAA_L20_c0.7", "caa", 20, 0.7), ("ActAdd_L14_c16", "actadd", 14, 16.0)]


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    methods, _ = build_methods(model, tok)

    xj = RESULTS / "xstest_gens_for_judge.json"; xd = json.loads(xj.read_text())
    hj = RESULTS / "harmbench_gens_for_judge.json"; hd = json.loads(hj.read_text())
    for key, m, layer, coef in CONFIGS:
        ctx = methods[m]["ctx"](layer, coef)
        with ctx("plain_gen"):
            xo = _generate(model, tok, [chat_wrap(tok, p) for p in xstest], GEN_TOKENS)
        xsm = sum(is_over_refusal(o) for o in xo) / len(xo)
        xd[key] = {"stringmatch_overrefusal": xsm,
                   "items": [{"prompt": p, "output": o} for p, o in zip(xstest, xo)]}
        with ctx("prefill_attack"):
            ho = _generate(model, tok, [chat_wrap(tok, p) + PREFILL for p in ood], GEN_TOKENS)
        conts = [PREFILL + o for o in ho]
        hsm = sum(is_refusal(c) for c in conts) / len(conts)
        hd[key] = {"stringmatch_refusal": hsm,
                   "items": [{"request": p, "continuation": c} for p, c in zip(ood, conts)]}
        print(f"{key}: XSTest over-refusal {xsm:.3f} | OOD refusal {hsm:.3f}", flush=True)
    xj.write_text(json.dumps(xd, ensure_ascii=False, indent=1))
    hj.write_text(json.dumps(hd, ensure_ascii=False, indent=1))
    print("saved gens", flush=True)


if __name__ == "__main__":
    main()
