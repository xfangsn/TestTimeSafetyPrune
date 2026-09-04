"""P-1 manipulation-check generation: for each pattern (uncertainty, backtracking) generate base vs
BLADE-G remove (α=0) vs amplify (α) traces on a FROZEN held-out prompt set (eval_messages, disjoint
from the direction/ELS TRAIN_MSGS). Saves per-item traces for blind Opus semantic annotation
(does the edit change the *semantic* pattern, not just keyword counts?). Runs on Hazel a10.
Usage: --model Qwen/Qwen3-1.7B --dirs qwen3_17b_dirs.pt --els reasoning_els_qwen3_17b.json --out manip_check_qwen3_17b.json
"""
import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.models import load_model
from ttsafety.hooks import get_decoder_layers
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_nll
from blade_refusal_amplify import scaled_weights
from ttsafety.weight_prune import pruned_weights
from reasoning_mask import build_mask, collect_Q

_STEER = os.environ.get("STEER_REPO")
sys.path.insert(0, str(Path(_STEER) / "messages") if _STEER
                else str(Path(__file__).resolve().parent / "steer_messages"))
from messages import messages as TRAIN_MSGS, eval_messages as EVAL_MSGS  # noqa: E402

RESULTS = Path("results"); MAX_NEW = 512


def parse_think(t):
    m = re.search(r"(.*?)</think>", t, re.DOTALL)
    return (m.group(1) if m else t).strip()


@torch.no_grad()
def gen(model, tok, prompts, bs=8):
    prev = tok.padding_side; tok.padding_side = "left"; outs = []
    try:
        for s in range(0, len(prompts), bs):
            enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--dirs", required=True)
    ap.add_argument("--els", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--behaviors", default="uncertainty-estimation,backtracking")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--rho", type=float, default=0.008)
    ap.add_argument("--amp-alphas", default="1.5"); ap.add_argument("--amp-rho", type=float, default=0.001)
    args = ap.parse_args()
    behaviors = args.behaviors.split(",")
    AMPS = [float(x) for x in args.amp_alphas.split(",")]

    model, tok = load_model(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    D = torch.load(RESULTS / args.dirs, weights_only=False)
    dirs, muC, muG = D["dirs"], D["muC"], D["muG"]
    assert D["model"] == args.model, f"dirs model {D['model']} != {args.model}"
    ELS = json.loads((RESULTS / args.els).read_text())["els"]
    allL = list(range(len(get_decoder_layers(model))))
    print("computing Q (g1scalar) on all layers ...", flush=True)
    Q, _ = collect_Q(model, tok, allL)

    train = {m["content"] for m in TRAIN_MSGS}
    held = [m for m in EVAL_MSGS if m["content"] not in train][:args.n]
    prompts = [tok.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in held]
    tasks = [m["content"] for m in held]
    print(f"held-out manipulation-check prompts: {len(held)}", flush=True)

    @contextmanager
    def noop():
        yield

    import math
    wiki = load_wikitext_text()
    base_nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=4000)

    out = {"model": args.model, "n": len(held), "rho": args.rho, "amp_alphas": AMPS,
           "behaviors": behaviors, "prov": {}, "ppl": {}, "items": []}
    # base once
    base_tr = [parse_think(g) for g in gen(model, tok, prompts)]
    for t, th in zip(tasks, base_tr):
        out["items"].append({"task": t, "behavior": "-", "mode": "base", "thinking": th,
                             "think_words": len(th.split())})
    for beh in behaviors:
        L = ELS[beh]["L_star"] or ELS[beh]["pool"][:1]
        conds = [("remove", 0.0, args.rho)] + [(f"amplify_a{a:g}", a, args.amp_rho) for a in AMPS]
        for mode, alpha, rho in conds:
            sel, prov = build_mask(model, dirs[beh], muC[beh], muG, L, Q=Q, alpha=alpha, rho=rho)
            out["prov"][f"{beh}:{mode}"] = prov
            cm = pruned_weights(model, sel) if mode == "remove" else scaled_weights(model, sel, alpha)
            with cm:
                trs = [parse_think(g) for g in gen(model, tok, prompts)]
                nll, _ = teacher_forced_nll(model, tok, wiki, max_tokens=4000)
            relppl = math.exp(nll - base_nll) - 1.0
            out["ppl"][f"{beh}:{mode}"] = relppl
            for t, th in zip(tasks, trs):
                out["items"].append({"task": t, "behavior": beh, "mode": mode, "thinking": th,
                                     "think_words": len(th.split())})
            print(f"  {beh} {mode} (L={L}, {prov['n_positive']} pos) done | Δppl {relppl:+.1%}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"saved results/{args.out} ({len(out['items'])} traces)", flush=True)


if __name__ == "__main__":
    main()
