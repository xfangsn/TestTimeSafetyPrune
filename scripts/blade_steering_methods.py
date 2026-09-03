"""Compare mainstream activation-steering methods for STRENGTHENING refusal, on the
base model, evaluated OOD (HarmBench jailbreak/prefill refusal) with WikiText ppl.
Methods (refusal direction = mean-diff, Arditi 2024):
  caa_1L    : add c*v at one mid layer, all positions           (CAA / ActAdd)
  caa_multi : add c*v_l at a band of mid layers                 (multi-layer CAA)
  addunit_1L: add c*vhat (unit) at one mid layer                (normalized ActAdd)
  clamp_1L  : set the projection onto vhat to target c at L12   (projection clamping)
Reference: BLADE weight-amplify (alpha sweep) from blade_amplify_steer_grid.json.
"""
import json
from contextlib import contextmanager
from pathlib import Path
import torch
from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import load_model
from blade_refusal_amplify import prefill_refusal

DATA = Path("data"); RESULTS = Path("results")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
N_DIR = 200; N_EVAL = 100; PPL_TOKENS = 5000; BAND = list(range(8, 17))
SEARCH_COEF = 0.2; SEARCH_PPL_BUDGET = 0.10; SEARCH_PPL_TOKENS = 2000


@contextmanager
def steer(model, layers, vraw, vunit, coef, mode):
    hs = []
    for li in layers:
        vr = vraw[li].to(model.device); vu = vunit[li].to(model.device)
        def mk(vr, vu):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                if mode == "add": h.add_(coef * vr.to(h.dtype))
                elif mode == "addunit": h.add_(coef * vu.to(h.dtype))
                elif mode == "clamp":
                    vv = vu.to(h.dtype); proj = (h * vv).sum(-1, keepdim=True)
                    h.add_((coef - proj) * vv)
                return out
            return hook
        hs.append(get_decoder_layers(model)[li].register_forward_hook(mk(vr, vu)))
    try: yield
    finally:
        for h in hs: h.remove()


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_train.jsonl")]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")]
    ood = [r["instruction"] for r in load_jsonl(DATA / "harmbench_standard.jsonl")][:N_EVAL]
    wiki = load_wikitext_text()
    vraw = extract_refusal_direction(model, tok, harmful[:N_DIR], harmless[:N_DIR])
    vunit = {l: (v.float() / v.float().norm().clamp_min(1e-8)) for l, v in vraw.items()}
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    base_ref = prefill_refusal(model, tok, ood)
    print(f"base OOD prefill-refusal {base_ref:.3f} | base ppl {base_ppl:.2f}", flush=True)

    # --- search best single layer for CAA (raw mean-diff add), fixed coef, ppl-budgeted ---
    n_layers = len(get_decoder_layers(model))
    base_ppl_s = teacher_forced_ppl(model, tok, wiki, max_tokens=SEARCH_PPL_TOKENS)
    search = []
    for li in range(n_layers):
        with steer(model, [li], vraw, vunit, SEARCH_COEF, "add"):
            r = prefill_refusal(model, tok, ood)
            p = teacher_forced_ppl(model, tok, wiki, max_tokens=SEARCH_PPL_TOKENS)
        dp = (p - base_ppl_s) / base_ppl_s
        search.append({"layer": li, "refusal": r, "ppl_delta": dp})
        print(f"  search L{li:<2} c={SEARCH_COEF} refusal {r:.3f}  Δppl {dp:+.1%}", flush=True)
    ok = [s for s in search if s["ppl_delta"] <= SEARCH_PPL_BUDGET]
    best = max(ok or search, key=lambda s: s["refusal"])
    L1 = best["layer"]
    print(f"==> best single layer L{L1} (refusal {best['refusal']:.3f}, "
          f"Δppl {best['ppl_delta']:+.1%}, budget {SEARCH_PPL_BUDGET:.0%})", flush=True)

    METHODS = {
        "caa_1L":     ([L1],  "add",     [0.1, 0.2, 0.3, 0.5]),
        "caa_multi":  (BAND,  "add",     [0.02, 0.05, 0.1, 0.2]),
        "addunit_1L": ([L1],  "addunit", [4.0, 8.0, 12.0, 16.0]),
        "clamp_1L":   ([L1],  "clamp",   [4.0, 8.0, 12.0, 20.0]),
    }
    out = {"base": {"refusal": base_ref, "ppl_delta": 0.0},
           "best_single_layer": L1, "layer_search": search, "methods": {}}
    for name, (layers, mode, coefs) in METHODS.items():
        rows = []
        for c in coefs:
            with steer(model, layers, vraw, vunit, c, mode):
                r = prefill_refusal(model, tok, ood)
                p = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            rows.append({"coef": c, "refusal": r, "ppl_delta": (p - base_ppl) / base_ppl})
            print(f"  {name:11} c={c:<5g} OOD refusal {r:.3f}  Δppl {(p-base_ppl)/base_ppl:+.1%}", flush=True)
        out["methods"][name] = rows

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_steering_methods.json").write_text(json.dumps(out, indent=2))
    print("saved results/blade_steering_methods.json", flush=True)


if __name__ == "__main__":
    main()
