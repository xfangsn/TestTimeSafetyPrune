"""Generate harmful_val outputs for LLM-judge re-evaluation (task #6) and
adversarial attacks (task #8).

Configs (greedy, max_new_tokens=128, same protocol as the sweeps):
  base, edge_s0.0005, ratio_s0.0001, signed_p0.0001_q0.0001, wei_p0.01_q0.01
Attacks on top:
  prefill "Sure, here is": base + edge_s0.0005
  decoding (temperature 1.0, top_p 0.9, seeds 0/1/2): edge_s0.0005

All records appended to data/judge_gens.jsonl (resume-safe per key).
Masks rebuilt via eval_downstream.build_selection (zero-checked).
"""

import json
from pathlib import Path

import torch

from eval_downstream import build_selection  # noqa: E402 (script dir import)
from ttsafety.data import load_jsonl
from ttsafety.models import chat_wrap, load_model
from ttsafety.weight_prune import pruned_weights

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "judge_gens.jsonl"
MAX_NEW_TOKENS = 128
CONFIGS = ["base", "edge_s0.0005", "ratio_s0.0001",
           "signed_p0.0001_q0.0001", "wei_p0.01_q0.01"]
PREFILL = "Sure, here is"
SAMPLE_SEEDS = (0, 1, 2)


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=MAX_NEW_TOKENS,
             do_sample=False, temperature=1.0, top_p=1.0,
             batch_size=16):
    """Batched generate over pre-formatted prompts; returns new tokens only."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            kwargs = {"do_sample": False}
            if do_sample:
                kwargs = {"do_sample": True, "temperature": temperature,
                          "top_p": top_p}
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, **kwargs,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            outputs.extend(tokenizer.batch_decode(new,
                                                  skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prev_side
    return outputs


def main():
    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["config"], r["attack"], r["index"]))

    model, tokenizer = load_model()
    val = load_jsonl(DATA / "harmful_val.jsonl")
    instructions = [r["instruction"] for r in val]
    prompts = [chat_wrap(tokenizer, s) for s in instructions]
    prefill_prompts = [p + PREFILL for p in prompts]

    selections = {}
    for config in CONFIGS:
        if config == "base":
            selections[config] = None
        else:
            selections[config] = build_selection(config)

    jobs = []  # (config, attack, prompts, sampling)
    for config in CONFIGS:
        jobs.append((config, "none", prompts, None))
    for config in ("base", "edge_s0.0005"):
        jobs.append((config, "prefill", prefill_prompts, None))
    for seed in SAMPLE_SEEDS:
        jobs.append(("edge_s0.0005", f"sampling_s{seed}", prompts,
                     {"temperature": 1.0, "top_p": 0.9, "seed": seed}))

    out_f = OUT.open("a", encoding="utf-8")
    for config, attack, job_prompts, sampling in jobs:
        todo = [i for i in range(len(instructions))
                if (config, attack, i) not in done]
        if not todo:
            print(f"skip {config}/{attack} (done)", flush=True)
            continue
        print(f"generating {config}/{attack} ({len(todo)} prompts) ...",
              flush=True)
        sel = selections[config]
        ctx = pruned_weights(model, sel) if sel is not None else _nullctx()
        with ctx:
            if sampling is None:
                outs = generate(model, tokenizer,
                                [job_prompts[i] for i in todo])
            else:
                torch.manual_seed(sampling["seed"])
                outs = generate(model, tokenizer,
                                [job_prompts[i] for i in todo],
                                do_sample=True,
                                temperature=sampling["temperature"],
                                top_p=sampling["top_p"])
        for i, out in zip(todo, outs):
            out_f.write(json.dumps({
                "config": config, "attack": attack, "index": i,
                "instruction": instructions[i], "output": out,
            }, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print(f"saved -> {OUT}", flush=True)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
