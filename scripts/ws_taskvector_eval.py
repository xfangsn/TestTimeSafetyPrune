"""Weight-steering (arXiv:2511.05408) sycophancy mitigation via task-vector
arithmetic, evaluated on OUR sycophancy A/B metric + WikiText ppl, for a
head-to-head with BLADE on Llama-3.2-3B.

task_vector = W(sycophant-FT) - W(non-sycophant-FT)
W_steered(scale) = W_base - scale * task_vector   (steer AWAY from sycophancy)
Sweep scale; report val pick-rate (side=matching) + ppl. scale=0 is base.
"""
import json
from pathlib import Path

import torch

from ttsafety.behaviors import fetch_ab, make_splits, pick_rate
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.models import env_info, load_model

DATA = Path("data"); RESULTS = Path("results"); FT = Path("data/ws_ft")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
PPL_TOKENS = 5000


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    syco = torch.load(FT / "llama32_syco.pt", map_location="cpu")
    nonsyco = torch.load(FT / "llama32_nonsyco.pt", map_location="cpu")
    # task vector (only where keys align and differ)
    tv = {}
    for k in base:
        if k in syco and k in nonsyco and syco[k].shape == base[k].shape:
            d = (syco[k].float() - nonsyco[k].float())
            if d.abs().sum() > 0:
                tv[k] = d
    print(f"task vector spans {len(tv)} tensors", flush=True)

    rows = fetch_ab("sycophancy", DATA / "behaviors")
    val = make_splits(rows)["val"]
    wiki = load_wikitext_text()

    base_ppl = None
    sweep = []
    for s in SCALES:
        with torch.no_grad():
            for k, d in tv.items():
                model.state_dict()[k].copy_((base[k].float() - s * d).to(base[k].dtype))
        pi, _ = pick_rate(model, tok, val, "matching")
        ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        if s == 0.0:
            base_ppl = ppl
        dppl = (ppl - base_ppl) / base_ppl
        sweep.append({"scale": s, "pick": pi, "ppl_delta": dppl})
        print(f"  scale={s:>4g} | sycophancy pick {pi:.3f} | Δppl {dppl:+.1%}", flush=True)
    # restore base
    with torch.no_grad():
        for k in tv:
            model.state_dict()[k].copy_(base[k])

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ws_taskvector_sycophancy.json").write_text(json.dumps(
        {"model": MODEL_ID, "method": "weight-steering task-vector",
         "base_ppl": base_ppl, "sweep": sweep, "env": env_info()}, indent=2))
    print("saved results/ws_taskvector_sycophancy.json", flush=True)


if __name__ == "__main__":
    main()
