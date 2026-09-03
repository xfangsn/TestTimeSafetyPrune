"""Generate the Refusal-Direction-addition (Arditi) frozen config (L16, c=0.35) outputs on XSTest
and HarmBench, append to the existing *_gens_for_judge.json under key 'RefDir_L16_c0.35'."""
import json
from pathlib import Path

from ttsafety.data import load_jsonl
from ttsafety.judge import is_over_refusal, is_refusal
from ttsafety.models import chat_wrap, load_model
from blade_steering_baselines import build_methods, _generate, GEN_TOKENS, PREFILL

DATA = Path("data"); RESULTS = Path("results")
KEY = "RefDir_L16_c0.5"; LAYER = 16; COEF = 0.5


def main():
    model, tok = load_model("meta-llama/Llama-3.2-3B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    xstest = [r["instruction"] for r in load_jsonl(DATA / "xstest_safe.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")]
    methods, _ = build_methods(model, tok)
    ctx = methods["arditi"]["ctx"](LAYER, COEF)

    # XSTest (plain gen)
    with ctx("plain_gen"):
        xo = _generate(model, tok, [chat_wrap(tok, p) for p in xstest], GEN_TOKENS)
    xsm = sum(is_over_refusal(o) for o in xo) / len(xo)
    xj = RESULTS / "xstest_gens_for_judge.json"; xd = json.loads(xj.read_text())
    xd[KEY] = {"stringmatch_overrefusal": xsm,
               "items": [{"prompt": p, "output": o} for p, o in zip(xstest, xo)]}
    xj.write_text(json.dumps(xd, ensure_ascii=False, indent=1))
    print(f"XSTest over-refusal {xsm:.3f}", flush=True)

    # HarmBench (prefill attack)
    with ctx("prefill_attack"):
        ho = _generate(model, tok, [chat_wrap(tok, p) + PREFILL for p in ood], GEN_TOKENS)
    conts = [PREFILL + o for o in ho]
    hsm = sum(is_refusal(c) for c in conts) / len(conts)
    hj = RESULTS / "harmbench_gens_for_judge.json"; hd = json.loads(hj.read_text())
    hd[KEY] = {"stringmatch_refusal": hsm,
               "items": [{"request": p, "continuation": c} for p, c in zip(ood, conts)]}
    hj.write_text(json.dumps(hd, ensure_ascii=False, indent=1))
    print(f"HarmBench OOD refusal {hsm:.3f}", flush=True)


if __name__ == "__main__":
    main()
