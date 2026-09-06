"""Does the amplify edit degenerate FREE GENERATION on generic WikiText (not just QA prompts)? Teacher-
forced ppl stays low under strong α, but free generation may collapse (exposure bias / error compounding).
We free-generate continuations from WikiText prefixes under base / α=0(remove) / α=4 / α=6 (same ρ=0.005
mask/L*), and report a lexical degeneration rate + samples + teacher-forced Δppl for reference.
Env: BLADE_MODEL. Usage: BLADE_MODEL=Qwen/Qwen3-8B .venv/bin/python scripts/measure_wikitext_gen.py"""
import json, os
from pathlib import Path
import torch
import ttsafety.extract as EX
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.extract import extract_refusal_direction
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from ttsafety.generic_importance import collect_c4_generic_importance
from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights
from blade_epistemic_els import qwen_wrap, last_token_moments, is_unc, COMPONENTS, PPL_TOKENS
from blade_epistemic_p0 import split_3way, score_fn_for
import blade_epistemic_p0 as P0

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
NPROMPT = 24; GEN_TOK = 128
RHO = float(os.environ.get("RHO", "0.005"))                 # ELS probe frac == final edit rho (matched)
WIKI_ALPHAS = [float(x) for x in os.environ.get("WIKI_ALPHAS", "4,6").split(",")]  # amplify alphas to test (plus base + remove)


def rep_score(t):
    w = t.split()
    if len(w) < 8: return 1.0 if len(w) < 3 else 0.0
    g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g)


@torch.no_grad()
def free_gen(model, tok, prompts, bs=8):
    prev = tok.padding_side; tok.padding_side = "left"; out = []
    try:
        for s in range(0, len(prompts), bs):
            enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True).to(model.device)
            g = model.generate(**enc, max_new_tokens=GEN_TOK, do_sample=False, pad_token_id=tok.pad_token_id)
            out.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return out


def main():
    P0.BLADE_G = True
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    EX.chat_wrap = qwen_wrap
    all_layers = list(range(len(get_decoder_layers(model))))
    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    tr, sel, _ = split_3way(rows)
    unc_tr = [r["question"] for r in tr if r["label"] == 1]; cert_tr = [r["question"] for r in tr if r["label"] == 0]
    unc_sel = [r["question"] for r in sel if r["label"] == 1]
    c4 = load_c4_text(); base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    directions = extract_refusal_direction(model, tok, unc_tr, cert_tr)
    muUNC = last_token_moments(model, tok, unc_tr, all_layers, COMPONENTS, qwen_wrap)
    muCERT = last_token_moments(model, tok, cert_tr, all_layers, COMPONENTS, qwen_wrap)
    print("Q ...", flush=True)
    P0.Q_GLOBAL, _ = collect_c4_generic_importance(model, tok, all_layers, COMPONENTS, text=c4, seqlen=2048,
                                                   batch_size=2, mode="g1scalar", max_tokens=65536)
    sfn = score_fn_for(model, directions, muUNC, muCERT, all_layers)
    def ppl_now(): return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    def measure():
        o = [x for x in unc_sel]; g = free_gen(model, tok, [qwen_wrap(tok, q) for q in o])
        return sum(is_unc(x) for x in g) / max(len(g), 1), ppl_now()
    ls_env = os.environ.get("L_STAR", "")
    if ls_env:                                  # skip ELS: reuse a known matched-rho L* (fast, avoids re-gen)
        L_star = [int(x) for x in ls_env.split(",")]
        print(f"RHO={RHO} L*={L_star} (from L_STAR env, ELS skipped)", flush=True)
    else:
        pool = solo_layer_pool(model, directions, muUNC, muCERT, all_layers, COMPONENTS, ppl_now, base_ppl,
                               screen_frac=RHO, beta=0.05, score_fn=sfn)   # matched: ELS probe frac == edit rho
        L_star = bestfirst_layers(model, directions, muUNC, muCERT, pool, COMPONENTS, measure, measure()[0],
                                  base_ppl, beta=0.05, eps=0.005, test_frac=RHO, score_fn=sfn)
        print(f"RHO={RHO} L*={L_star}", flush=True)
    selw = selection_from_ranking(rank_weight_indices(sfn(model, directions, muUNC, muCERT, L_star, COMPONENTS), RHO + 0.01), RHO)

    # WikiText prefixes (first ~40 words of each of NPROMPT chunks)
    wiki = load_wikitext_text()
    chunks = [c.strip() for c in wiki.split("\n") if len(c.split()) > 60][:NPROMPT]
    prompts = [" ".join(c.split()[:40]) for c in chunks]
    print(f"{len(prompts)} WikiText prefixes; base ppl {base_ppl:.2f}", flush=True)

    import contextlib
    conds = [("base", contextlib.nullcontext()), (f"r{RHO}_alpha0_remove", pruned_weights(model, selw))]
    for a in WIKI_ALPHAS:
        conds.append((f"r{RHO}_alpha{a}", scaled_weights(model, selw, a)))
    report = {"model": MODEL_ID, "rho": RHO, "L_star": L_star, "env": env_info(), "conds": {}}
    for label, cm in conds:
        with cm:
            outs = free_gen(model, tok, prompts)
            pc = teacher_forced_ppl(model, tok, load_wikitext_text(), max_tokens=PPL_TOKENS)
        dg = sum(rep_score(o) > 0.5 for o in outs) / len(outs)
        report["conds"][label] = {"wiki_free_degen": dg, "wiki_ppl": pc,
                                  "samples": [o[:160] for o in outs[:4]]}
        print(f"  {label:18} WikiText free-gen degen {dg:.2f}  wiki_ppl {pc:.2f}", flush=True)
        for o in outs[:2]:
            print(f"      | {o[:120]}".replace(chr(10), ' '), flush=True)
    tag = os.environ.get("OUT_TAG", "")
    (RESULTS / f"wikitext_gen{tag}_qwen3-8b.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved results/wikitext_gen{tag}_qwen3-8b.json", flush=True)


if __name__ == "__main__":
    main()
