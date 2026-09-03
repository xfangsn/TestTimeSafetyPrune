"""Generate Qwen3-4B thinking traces on the repo task set (exploratory, since Qwen3-4B has no bundled
annotations). Saves results/qwen3_traces.json = [{task, full, thinking}]. Directions will be built
from KEYWORD-matched spans (annotation-free proxy), so this only needs the raw traces."""
import json
import re
import sys
from pathlib import Path

import torch

from ttsafety.models import load_model

import os
_STEER = os.environ.get("STEER_REPO")
sys.path.insert(0, str(Path(_STEER) / "messages") if _STEER
                else str(Path(__file__).resolve().parent / "steer_messages"))
from messages import messages as TRAIN  # noqa: E402

RESULTS = Path("results")
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument("--model",default="Qwen/Qwen3-4B")
_ap.add_argument("--out",default="qwen3_traces.json")
_ap.add_argument("--n",type=int,default=200)
_A=_ap.parse_args()
N=_A.n; MAX_NEW=1024


@torch.no_grad()
def gen(model, tok, prompts, bs=16):
    prev = tok.padding_side; tok.padding_side = "left"; outs = []
    try:
        for s in range(0, len(prompts), bs):
            enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=False))
            print(f"  {min(s+bs,len(prompts))}/{len(prompts)}", flush=True)
    finally:
        tok.padding_side = prev
    return outs


def main():
    model, tok = load_model(_A.model)
    tasks = [m["content"] for m in TRAIN[:N]]
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                       add_generation_prompt=True) for t in tasks]
    outs = gen(model, tok, prompts)
    data = []
    n_closed = 0
    for t, o in zip(tasks, outs):
        # full assistant text starts after the generation prompt; strip trailing special tokens
        full = o
        m = re.search(r"<think>(.*?)</think>", "<think>" + o, re.DOTALL)
        thinking = (m.group(1).strip() if m else o.split("</think>")[0].strip())
        if "</think>" in o:
            n_closed += 1
        data.append({"task": t, "full": full, "thinking": thinking})
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / _A.out).write_text(json.dumps(data, ensure_ascii=False, indent=1))
    wl = sum(len(d["thinking"].split()) for d in data) / len(data)
    print(f"\nsaved results/{_A.out} | {len(data)} traces | closed </think> {n_closed}/{len(data)} "
          f"| mean thinking words {wl:.0f}", flush=True)


if __name__ == "__main__":
    main()
