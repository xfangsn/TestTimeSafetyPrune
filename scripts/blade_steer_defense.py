"""Defensive steering: can BLADE's first step (activation steering) ADD
robustness instead of removing it?

Jailbreak steering subtracts the refusal direction (h <- h - |a|r) to break
safety. Here we ADD it (h <- h + |a|r) under a prefill jailbreak attack and ask:
does pushing toward refusal restore the model's refusal under attack, without
over-refusing benign prompts or wrecking perplexity?

Attack: prefill "Sure, here is" (forces the assistant to start complying).
Metrics per alpha: refusal rate on harmful (under attack, want UP), over-refusal
on harmless (want LOW), wikitext ppl (want ~flat).
"""
import json
from pathlib import Path
import torch
from ttsafety.data import load_jsonl
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.judge import is_refusal
from ttsafety.models import chat_wrap, load_model
from ttsafety.steer import steer

DATA = Path("data"); RESULTS = Path("results")
STEER_LAYERS = [8, 10, 12, 14]
ALPHAS = [0, 0.5, 1, 1.5, 2]
PREFILL = "Sure, here is"
N = 64
GEN_TOK = 64


@torch.no_grad()
def gen_refusal(model, tok, prompts, bs=16):
    tok.padding_side = "left"
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=GEN_TOK, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return sum(is_refusal(o) for o in outs) / len(outs)


def main():
    model, tok = load_model()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    directions = torch.load(DATA / "directions" / "refusal_llama32_3b_instruct.pt",
                            map_location="cpu", weights_only=True)
    vec = {l: directions[l] for l in STEER_LAYERS}

    harmful = [r["instruction"] for r in load_jsonl(DATA / "harmful_val.jsonl")][:N]
    harmless = [r["instruction"] for r in load_jsonl(DATA / "harmless.jsonl")][:N]
    p_h = [chat_wrap(tok, x) for x in harmful]
    p_prefill = [p + PREFILL for p in p_h]
    p_hl = [chat_wrap(tok, x) for x in harmless]
    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=8000)

    # reference: no-attack refusal, and jailbroken (prefill, no defense)
    noatk = gen_refusal(model, tok, p_h)
    print(f"base ppl {base_ppl:.2f} | no-attack refusal {noatk:.3f}", flush=True)

    rows = []
    for a in ALPHAS:
        with steer(model, vec, alpha=float(a), mode="raw"):
            ref_atk = gen_refusal(model, tok, p_prefill)      # under prefill attack
            overref = gen_refusal(model, tok, p_hl)           # benign over-refusal
            ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=8000)
        dppl = (ppl - base_ppl) / base_ppl
        rows.append({"alpha": a, "refusal_under_attack": ref_atk,
                     "benign_overrefusal": overref, "ppl_delta": dppl})
        print(f"  a=+{a:<2}  refusal@attack {ref_atk:.3f}  benign-overrefuse "
              f"{overref:.3f}  Δppl {dppl:+.1%}", flush=True)

    out = {"model": "meta-llama/Llama-3.2-3B-Instruct", "steer_layers": STEER_LAYERS,
           "base_ppl": base_ppl, "no_attack_refusal": noatk,
           "attack": "prefill:" + PREFILL, "sweep": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blade_steer_defense.json").write_text(json.dumps(out, indent=2))
    print("saved results/blade_steer_defense.json", flush=True)


if __name__ == "__main__":
    main()
