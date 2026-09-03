"""Regenerate held-out thinking traces for the key BLADE-reasoning configs and SAVE the raw text,
so a Fable-5.1 annotator can validate the keyword-proxy behavior rates. Configs:
  base | uncertainty BLADE-remove (α=0) | uncertainty RANDOM-remove (α=0) | backtracking BLADE-remove.
Writes results/reasoning_eval_traces.json = {config: [{task, thinking}]}."""
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.models import load_model
from ttsafety.sycophancy import score_edges
from ttsafety.weight_prune import random_scores_like, rank_weight_indices, selection_from_ranking
from blade_refusal_amplify import scaled_weights

REPO = Path("/tmp/claude-1000/-home-xfang1999-Projects-TestTimeSafetyPrune/"
            "e16f646c-64a5-440b-bd68-985c068d25df/scratchpad/steering-thinking-llms")
sys.path.insert(0, str(REPO / "messages"))
from messages import messages as TRAIN_MSGS, eval_messages as EVAL_MSGS  # noqa: E402

RESULTS = Path("results")
LAYERS = [15, 16, 17, 18]; RHO = 0.01; SCREEN = 0.03; MAX_NEW = 512


def parse_think(text):
    m = re.search(r"(.*?)</think>", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


@torch.no_grad()
def generate(model, tok, prompts, bs=12):
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


def main():
    model, tok = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    D = torch.load(RESULTS / "reasoning_dirs.pt", weights_only=False)
    dirs, muC, muG = D["dirs"], D["muC"], D["muG"]
    train_set = {m["content"] for m in TRAIN_MSGS}
    evals = [m for m in EVAL_MSGS if m["content"] not in train_set]
    prompts = [tok.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in evals]
    tasks = [m["content"] for m in evals]

    def mask(beh, seed=None):
        scores = score_edges(model, dirs[beh], muC[beh], muG, LAYERS, "both")
        if seed is not None:
            scores = random_scores_like(scores, seed=seed)
        return selection_from_ranking(rank_weight_indices(scores, SCREEN), RHO)

    unc_blade = mask("uncertainty-estimation")
    unc_rand = mask("uncertainty-estimation", seed=7)
    back_blade = mask("backtracking")

    @contextmanager
    def noop():
        yield

    def wcm(sel):
        @contextmanager
        def cm():
            with scaled_weights(model, sel, 0.0):
                yield
        return cm()

    configs = {"base": noop(), "unc_blade_remove": wcm(unc_blade),
               "unc_random_remove": wcm(unc_rand), "back_blade_remove": wcm(back_blade)}
    out = {}
    for name, cm in configs.items():
        with cm:
            gens = generate(model, tok, prompts)
        out[name] = [{"task": t, "thinking": parse_think(g)} for t, g in zip(tasks, gens)]
        print(f"{name}: {len(out[name])} traces, mean words "
              f"{sum(len(x['thinking'].split()) for x in out[name])/len(out[name]):.0f}", flush=True)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "reasoning_eval_traces.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved results/reasoning_eval_traces.json", flush=True)


if __name__ == "__main__":
    main()
