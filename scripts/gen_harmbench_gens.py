"""Generate HarmBench standard-behavior outputs for the held-out evaluation.

One-time evaluation per docs/plan-harmbench-heldout.md. Five locked configs
(greedy, max_new_tokens=128, same protocol as the sweeps):
  base, edge_s0.0005, ratio_s0.0001, signed_p0.0001_q0.0001, wei_p0.01_q0.01

Records appended to data/harmbench_gens.jsonl (resume-safe per (config, index)).
Masks rebuilt via eval_downstream.build_selection; each pruned config is
verified selected-values-zero before generation.
"""

import json
from pathlib import Path

from eval_downstream import build_selection, verify_zeroed  # noqa: E402
from gen_adversarial_gens import generate, _nullctx  # noqa: E402
from ttsafety.data import load_jsonl
from ttsafety.models import chat_wrap, load_model
from ttsafety.weight_prune import pruned_weights

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "harmbench_gens.jsonl"
CONFIGS = ["base", "edge_s0.0005", "ratio_s0.0001",
           "signed_p0.0001_q0.0001", "wei_p0.01_q0.01"]


def main():
    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["config"], r["index"]))

    model, tokenizer = load_model()
    rows = load_jsonl(DATA / "harmbench_standard.jsonl")
    instructions = [r["instruction"] for r in rows]
    prompts = [chat_wrap(tokenizer, s) for s in instructions]

    out_f = OUT.open("a", encoding="utf-8")
    for config in CONFIGS:
        todo = [i for i in range(len(instructions))
                if (config, i) not in done]
        if not todo:
            print(f"skip {config} (done)", flush=True)
            continue
        print(f"generating {config} ({len(todo)} prompts) ...", flush=True)
        sel = build_selection(config)
        ctx = pruned_weights(model, sel) if sel is not None else _nullctx()
        with ctx:
            if sel is not None:
                checked = verify_zeroed(model, sel)
                print(f"  {config}: verified {checked} selected entries zeroed",
                      flush=True)
            outs = generate(model, tokenizer, [prompts[i] for i in todo])
        for i, out in zip(todo, outs):
            out_f.write(json.dumps({
                "config": config, "index": i,
                "instruction": instructions[i], "output": out,
            }, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
