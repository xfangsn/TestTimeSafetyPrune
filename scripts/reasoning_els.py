"""BLADE Effective-Layer Selection (ELS) for reasoning behaviors — AUTO-selects the effective
layer(s) per behavior under a perplexity budget, instead of using the paper's hand-picked attribution
layers. Reuses the exact refusal ELS machinery (solo_layer_pool -> bestfirst_layers) with the metric
swapped to the behavior's keyword rate on a screen set (lower = more removed). Writes
results/reasoning_els_<key>.json = {behavior: {"L_star": [...], "pool": [...]}}.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

from ttsafety.behaviors import bestfirst_layers, solo_layer_pool
from ttsafety.eval import load_c4_text, load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model

import os
_STEER = os.environ.get("STEER_REPO")
sys.path.insert(0, str(Path(_STEER) / "messages") if _STEER
                else str(Path(__file__).resolve().parent / "steer_messages"))
from messages import messages as TRAIN_MSGS  # noqa: E402

RESULTS = Path("results")
COMPONENTS = "both"
BETA = 0.05          # ppl budget (C4)
EPS = 0.5            # min keyword-rate improvement per added layer (per 1000 words)
SCREEN_FRAC = 0.008  # solo-pool prune fraction
TEST_FRAC = 0.008    # best-first joint prune fraction
N_SCREEN = 16        # screen tasks for the behavior metric
PPL_TOKENS = 4000
MAX_NEW = 384
KW = {
    "uncertainty-estimation": [" maybe", " perhaps", "not sure", "i think", " possibly", " might ",
                               "could be", "i'm not", " unsure", " i guess", "not certain"],
    "example-testing": ["for example", "for instance", "let's try", "e.g.", "let me test",
                        " suppose ", "let's test", " let me try"],
    "backtracking": [" wait", " actually", "reconsider", " hmm", "scratch that",
                     "on second thought", " no,", "let me re", " but wait"],
    "adding-knowledge": ["i know that", "i remember", "recall that", "it's known", "the formula",
                         "by definition", "in general,"],
}


def parse_think(text):
    m = re.search(r"(.*?)</think>", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


@torch.no_grad()
def gen(model, tok, prompts, bs=8):
    prev = tok.padding_side; tok.padding_side = "left"
    outs = []
    try:
        for s in range(0, len(prompts), bs):
            enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return outs


def kw_rate(model, tok, prompts, beh):
    thinks = [parse_think(o) for o in gen(model, tok, prompts)]
    per = [1000.0 * sum(t.lower().count(k) for k in KW[beh]) / max(1, len(t.split())) for t in thinks]
    return sum(per) / len(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="reasoning_dirs_llama8b.pt")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--key", default="llama8b")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--screen-frac", type=float, default=SCREEN_FRAC)
    args = ap.parse_args()
    eps = args.eps; test_frac = args.test_frac; screen_frac = args.screen_frac
    print(f"ELS params: eps={eps} test_frac={test_frac} screen_frac={screen_frac} beta={BETA}", flush=True)

    model, tok = load_model(args.model)
    D = torch.load(RESULTS / args.dirs, weights_only=False)
    dirs, muC, muG = D["dirs"], D["muC"], D["muG"]
    behaviors = D["behaviors"]
    all_layers = list(range(len(get_decoder_layers(model))))
    screen = [tok.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
              for m in TRAIN_MSGS[:N_SCREEN]]
    c4 = load_c4_text(); wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    base_ppl_wiki = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
    print(f"{args.model}: {len(all_layers)} layers | base C4 ppl {base_ppl:.2f}", flush=True)

    def ppl_now():
        return teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)

    out = {"model": args.model, "beta": BETA, "eps": EPS, "n_screen": N_SCREEN,
           "base_ppl_c4": base_ppl, "base_ppl_wiki": base_ppl_wiki, "els": {}, "env": env_info()}
    for beh in behaviors:
        base_metric = kw_rate(model, tok, screen, beh)
        print(f"\n== ELS {beh} | base rate {base_metric:.2f} ==", flush=True)
        pool = solo_layer_pool(model, dirs[beh], muC[beh], muG, all_layers, COMPONENTS,
                               ppl_now, base_ppl, screen_frac=screen_frac, beta=BETA)
        print(f"  pool (within ppl budget): {pool}", flush=True)

        def measure(beh=beh):
            return kw_rate(model, tok, screen, beh), ppl_now()

        L_star = bestfirst_layers(model, dirs[beh], muC[beh], muG, pool, COMPONENTS,
                                  measure, base_metric, base_ppl, beta=BETA, eps=eps,
                                  test_frac=test_frac)
        print(f"  -> L* = {L_star}", flush=True)
        out["els"][beh] = {"L_star": L_star, "pool": pool, "base_rate": base_metric}
        (RESULTS / f"reasoning_els_{args.key}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved results/reasoning_els_{args.key}.json", flush=True)


if __name__ == "__main__":
    main()
