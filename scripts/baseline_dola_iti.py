"""Baselines for the hallucination-reduction comparison: DoLa (Chuang et al. 2309.03883, HF-native
dola_layers) and ITI (Li et al. 2306.03341, per-head truthful-direction steering). Both aim to reduce
hallucination / raise truthful commitment; we compare them to our BLADE weight edit on the SAME
closed-book OOD prompts (SelfAware unans/ans + FalseQA false/true) and the SAME kimi judge.

ITI is fit on OUR certain/uncertain contrast (epistemic_pairs_v2) at the last prompt token — same signal
as BLADE — so the comparison isolates the MECHANISM (head activation-steering vs sparse weight edit),
not the direction source. Steering pushes toward the 'uncertain/abstain' direction (reduce hallucination).

Env: BLADE_MODEL. Usage: BLADE_MODEL=Qwen/Qwen3-8B .venv/bin/python scripts/baseline_dola_iti.py
Saves results/baseline_dola_iti_<tag>.json (generations for kimi judge) + ITI provenance.
"""
import csv
import json
import os
from pathlib import Path

import torch

from ttsafety.eval import load_c4_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.models import env_info, load_model
from blade_epistemic_els import qwen_wrap, PPL_TOKENS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"; DATA = ROOT / "data"
MODEL_ID = os.environ.get("BLADE_MODEL", "Qwen/Qwen3-8B")
GEN_TOK = 128
CAP = 70
ITI_K = int(os.environ.get("ITI_K", "48"))          # top heads (ITI paper ~48)
ITI_ALPHAS = [float(x) for x in os.environ.get("ITI_ALPHAS", "2,6,10").split(",")]  # in units of proj-std


def load_ood():
    fq = list(csv.DictReader(open(DATA / "abstention" / "falseqa_test.csv")))
    out = []
    for q in [r["question"] for r in fq if r["label"] == "1"][:CAP]:
        out.append({"dataset": "falseqa", "gold": "false_premise", "question": q})
    for q in [r["question"] for r in fq if r["label"] == "0"][:CAP]:
        out.append({"dataset": "falseqa", "gold": "true_premise", "question": q})
    ex = json.loads((DATA / "abstention" / "SelfAware.json").read_text())["example"]
    for q in [x["question"] for x in ex if not x["answerable"]][:CAP]:
        out.append({"dataset": "selfaware", "gold": "unanswerable", "question": q})
    for q in [x["question"] for x in ex if x["answerable"]][:CAP]:
        out.append({"dataset": "selfaware", "gold": "answerable", "question": q})
    return out


@torch.no_grad()
def head_acts(model, tok, prompts, blocks, nh, hd, bs=8):
    """mean o_proj-input per head at the last prompt token -> {layer: (nh, hd)} averaged over prompts."""
    acc = {i: torch.zeros(nh, hd) for i in range(len(blocks))}
    st = {}

    def mk(i):
        def hook(_m, args):
            v = args[0].float()                       # (B,T, nh*hd)
            rows = torch.arange(v.shape[0])
            last = v[rows, st["last"]].view(v.shape[0], nh, hd)  # (B,nh,hd)
            st["b"][i] = last.sum(0).cpu()
        return hook

    hs = [blocks[i].self_attn.o_proj.register_forward_pre_hook(mk(i)) for i in range(len(blocks))]
    n = 0
    try:
        for s in range(0, len(prompts), bs):
            texts = [qwen_wrap(tok, p) for p in prompts[s:s + bs]]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="right",
                      add_special_tokens=False).to(model.device)
            st["last"] = enc["attention_mask"].sum(1) - 1; st["b"] = {}
            model(**enc, use_cache=False)
            for i in acc:
                acc[i] += st["b"][i]
            n += enc["input_ids"].shape[0]
    finally:
        for h in hs:
            h.remove()
    return {i: acc[i] / n for i in acc}, n


@torch.no_grad()
def proj_std(model, tok, prompts, blocks, nh, hd, dirs_unit, bs=8):
    """std of <head_act, unit_dir> at the last prompt token, per (layer,head) in dirs_unit — the ITI dose unit."""
    sums = {k: [0.0, 0.0, 0] for k in dirs_unit}   # sum, sumsq, n
    st = {}

    def mk(i):
        def hook(_m, args):
            v = args[0].float()
            rows = torch.arange(v.shape[0])
            st["b"][i] = v[rows, st["last"]].view(v.shape[0], nh, hd).cpu()   # (B,nh,hd)
        return hook

    hs = [blocks[i].self_attn.o_proj.register_forward_pre_hook(mk(i)) for i in {k[0] for k in dirs_unit}]
    try:
        for s in range(0, len(prompts), bs):
            texts = [qwen_wrap(tok, p) for p in prompts[s:s + bs]]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="right",
                      add_special_tokens=False).to(model.device)
            st["last"] = enc["attention_mask"].sum(1) - 1; st["b"] = {}
            model(**enc, use_cache=False)
            for (i, h) in dirs_unit:
                p = (st["b"][i][:, h] @ dirs_unit[(i, h)])   # (B,)
                sums[(i, h)][0] += float(p.sum()); sums[(i, h)][1] += float((p * p).sum()); sums[(i, h)][2] += p.numel()
    finally:
        for x in hs:
            x.remove()
    out = {}
    for k, (s1, s2, n) in sums.items():
        out[k] = max((s2 / n - (s1 / n) ** 2), 1e-8) ** 0.5
    return out


@torch.no_grad()
def gen_plain(model, tok, prompts, bs=12):
    prev = tok.padding_side; tok.padding_side = "left"; out = []
    try:
        for s in range(0, len(prompts), bs):
            texts = [qwen_wrap(tok, p) for p in prompts[s:s + bs]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=GEN_TOK, do_sample=False, pad_token_id=tok.pad_token_id)
            out.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return out


@torch.no_grad()
def gen_dola(model, tok, prompts, bucket="high", alpha=0.1, rep=1.2, bs=8):
    """Manual DoLa (Chuang et al. 2309.03883): contrast the mature (final) layer against an adaptively
    selected premature layer (max JS-divergence within a high/low bucket), with the adaptive plausibility
    constraint (APC) + repetition penalty. HF's dola_layers is broken on tf5.15 (community module)."""
    blocks = get_decoder_layers(model); N = len(blocks)
    cand = list(range(N // 2, N - 1, 2)) if bucket == "high" else list(range(0, N // 2, 2))
    norm = model.model.norm; head = model.lm_head
    prev = tok.padding_side; tok.padding_side = "left"; outs = []
    for s in range(0, len(prompts), bs):
        texts = [qwen_wrap(tok, p) for p in prompts[s:s + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        B = ids.shape[0]; past = None
        gen = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        cur, cur_attn = ids, attn
        for step in range(GEN_TOK):
            o = model(input_ids=cur if past is None else cur[:, -1:], attention_mask=cur_attn,
                      past_key_values=past, use_cache=True, output_hidden_states=True)
            past = o.past_key_values
            hs = o.hidden_states
            mat_lp = torch.log_softmax(head(norm(hs[-1][:, -1])).float(), -1)   # (B,V)
            mat_p = mat_lp.exp()
            best_jsd = torch.full((B,), -1.0, device=ids.device); prem_lp = mat_lp.clone()
            for l in cand:
                plp = torch.log_softmax(head(norm(hs[l][:, -1])).float(), -1)
                m = 0.5 * (mat_p + plp.exp())
                jsd = 0.5 * (mat_p * (mat_lp - m.log())).sum(-1) + 0.5 * (plp.exp() * (plp - m.log())).sum(-1)
                pick = jsd > best_jsd
                best_jsd = torch.where(pick, jsd, best_jsd)
                prem_lp = torch.where(pick[:, None], plp, prem_lp)
            logits = mat_lp - prem_lp                                           # DoLa contrast
            thresh = alpha * mat_p.max(-1, keepdim=True).values                 # APC (plausibility)
            logits = logits.masked_fill(mat_p < thresh, -1e9)
            # restrict contrast to the top-50 mature tokens (kills rare-token promotion / garbage)
            topk = mat_lp.topk(50, dim=-1).indices
            keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk, True)
            logits = logits.masked_fill(~keep, -1e9)
            for b in range(B):                                                  # repetition penalty
                for t in set(gen[b]):
                    logits[b, t] /= rep
                if len(gen[b]) >= 2:                                            # block repeated 3-grams
                    a2, a1 = gen[b][-2], gen[b][-1]
                    for k in range(len(gen[b]) - 2):
                        if gen[b][k] == a2 and gen[b][k + 1] == a1:
                            logits[b, gen[b][k + 2]] = -1e9
            nxt = logits.argmax(-1)                                             # greedy
            nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
            for b in range(B):
                if not done[b]:
                    gen[b].append(int(nxt[b]))
            done = done | (nxt == tok.eos_token_id)
            cur = nxt[:, None]
            cur_attn = torch.cat([cur_attn, (~done)[:, None].long()], dim=1)
            if done.all():
                break
        outs.extend(tok.decode([t for t in g if t != tok.pad_token_id], skip_special_tokens=True) for g in gen)
    tok.padding_side = prev
    return outs


@torch.no_grad()
def gen_iti(model, tok, prompts, blocks, add, bs=12):
    """add[layer] = (nh*hd,) steering added to o_proj input (decode positions)."""
    handles = []

    def mk(i):
        vec = add[i]
        def hook(_m, args):
            x = args[0]
            if x.shape[1] > 1:      # decode-only (skip prefill)
                return None
            return (x + vec.to(x.dtype).to(x.device),) + tuple(args[1:])
        return hook
    for i in range(len(blocks)):
        handles.append(blocks[i].self_attn.o_proj.register_forward_pre_hook(mk(i)))
    try:
        return gen_plain(model, tok, prompts)
    finally:
        for h in handles:
            h.remove()


def main():
    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    blocks = get_decoder_layers(model)
    cfg = model.config
    nh, hd = cfg.num_attention_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    rows = json.loads((RESULTS / "epistemic_pairs_v2.json").read_text())["rows"]
    unc = [r["question"] for r in rows if r["label"] == 1]
    cert = [r["question"] for r in rows if r["label"] == 0]
    print("fitting ITI head directions (uncertain vs certain, last prompt token) ...", flush=True)
    mu_u, _ = head_acts(model, tok, unc, blocks, nh, hd)
    mu_c, _ = head_acts(model, tok, cert, blocks, nh, hd)
    # per-head separability (L2 of mean-diff) -> pick top-K heads; direction = unit mean-diff; sigma = |diff|
    diffs = {(i, h): (mu_u[i][h] - mu_c[i][h]) for i in range(len(blocks)) for h in range(nh)}
    ranked = sorted(diffs, key=lambda k: -diffs[k].norm().item())[:ITI_K]
    dirs = {k: diffs[k] / diffs[k].norm().clamp_min(1e-6) for k in ranked}
    print(f"ITI top-{ITI_K} heads across layers {sorted({i for i,_ in ranked})}; computing proj-std ...", flush=True)
    sigma = proj_std(model, tok, unc + cert, blocks, nh, hd, dirs)   # ITI dose unit = std of projection

    def add_vec(alpha):
        add = {i: torch.zeros(nh * hd) for i in range(len(blocks))}
        for (i, h) in ranked:
            add[i].view(nh, hd)[h] = alpha * sigma[(i, h)] * dirs[(i, h)]
        return add

    items = load_ood()
    prompts = [it["question"] for it in items]
    c4 = load_c4_text(); base_ppl = teacher_forced_ppl(model, tok, c4, max_tokens=PPL_TOKENS)
    print(f"OOD items {len(items)}; base ppl {base_ppl:.2f}", flush=True)

    gens = {"base": gen_plain(model, tok, prompts)}
    print("  base done", flush=True)
    if not os.environ.get("SKIP_DOLA"):
        gens["dola_high"] = gen_dola(model, tok, prompts, bucket="high"); print("  dola_high done", flush=True)
        gens["dola_low"] = gen_dola(model, tok, prompts, bucket="low"); print("  dola_low done", flush=True)
    ppl = {"base": 0.0}
    for a in ITI_ALPHAS:
        add = add_vec(a)
        gens[f"iti_a{a}"] = gen_iti(model, tok, prompts, blocks, add)
        # ppl under the ITI hook (steering all positions for ppl is fine; use decode-style: apply to all)
        print(f"  iti_a{a} done", flush=True)

    report = {"model": MODEL_ID, "iti_k": ITI_K, "iti_alphas": ITI_ALPHAS, "iti_heads": [list(k) for k in ranked],
              "base_ppl_c4": base_ppl, "env": env_info(), "conditions": list(gens), "items": []}
    for i, it in enumerate(items):
        report["items"].append({**it, **{c: gens[c][i] for c in gens}})
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"baseline_dola_iti_{tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved results/baseline_dola_iti_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
