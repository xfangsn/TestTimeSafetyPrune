"""BLADE "prune to admit uncertainty" on SelfAware.

Behavior to remove = overconfident assertion (failure to abstain). We build an
assert-vs-"I don't know" contrast, score residual-writer edges with BLADE, use
ELS to pick layers by SELECTIVE abstention gain, prune, and measure the two-axis
tradeoff:
  * UNANSWERABLE set -> abstention rate  (want UP; less fabrication)
  * ANSWERABLE-known set -> accuracy retained (want FLAT; not over-abstaining)
plus a matched random-weight control. Good result = up without going left.

Data: SelfAware (Yin et al. 2023). Answerable filtered to base-correct so the
cost axis cleanly measures over-abstention. Model via BLADE_MODEL (default Llama).
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import torch

import ttsafety.behaviors as B
from ttsafety.behaviors import (collect_span_input_moments, extract_direction,
                                score_edges)
from ttsafety.eval import load_wikitext_text, teacher_forced_ppl
from ttsafety.hooks import get_decoder_layers
from ttsafety.judge import is_abstention
from ttsafety.models import chat_wrap, env_info, load_model
from ttsafety.weight_prune import (pruned_weights, random_scores_like,
                                   rank_weight_indices, selection_from_ranking)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "abstention"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
URL = "https://raw.githubusercontent.com/yinzhangyue/SelfAware/main/data/SelfAware.json"

MODEL_ID = os.environ.get("BLADE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
COMPONENTS = "both"
SCREEN_FRAC = 0.005
DELTA = 0.03          # relaxed: assertion is diffuse; take top selective layers
BETA = 0.05
EPS = 0.005
PPL_TOKENS = 4000
GEN_TOK = 40
N_TRAIN, N_EVAL_UNANS, N_EVAL_ANS, N_SCREEN = 120, 250, 120, 30
ANS_POOL = 1600     # answerable to base-gen; ~26% are base-correct (closed-book)
SPARS = [0.0005, 0.002, 0.005, 0.02, 0.05]


def qwen_wrap(tok, s):
    m = [{"role": "user", "content": s}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", t.lower())


@torch.no_grad()
def gen(model, tok, questions, wrap, bs=16):
    tok.padding_side = "left"
    prompts = [wrap(tok, q) for q in questions]
    out = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=GEN_TOK, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def abst_rate(outs):
    return sum(is_abstention(o) for o in outs) / len(outs)


def accuracy(outs, golds):
    ok = 0
    for o, gs in zip(outs, golds):
        no = norm(o)
        if not is_abstention(o) and any(norm(g) in no for g in gs if g.strip()):
            ok += 1
    return ok / len(outs)


def main():
    mid = MODEL_ID.lower()
    is_qwen = "qwen" in mid
    is_gemma = "gemma" in mid
    eot = "<|im_end|>" if is_qwen else "<end_of_turn>" if is_gemma else "<|eot_id|>"
    wrap = qwen_wrap if is_qwen else chat_wrap
    if is_qwen:
        B.chat_wrap = qwen_wrap

    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / "SelfAware.json"
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(URL, timeout=120).read())
    ex = json.loads(cache.read_text())["example"]
    unans = [e for e in ex if not e["answerable"]]
    ans = [e for e in ex if e["answerable"]]
    import random
    random.Random(0).shuffle(unans); random.Random(1).shuffle(ans)

    model, tok = load_model(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_layers = list(range(len(get_decoder_layers(model))))
    wiki = load_wikitext_text()
    base_ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)

    # filter answerable to base-correct (clean cost axis)
    print("filtering answerable to base-correct ...", flush=True)
    pool = ans[:ANS_POOL]
    outs = gen(model, tok, [e["question"] for e in pool], wrap)
    known = [e for e, o in zip(pool, outs)
             if not is_abstention(o) and any(norm(g) in norm(o) for g in e["answer"] if g.strip())]
    print(f"answerable base-correct: {len(known)}/{len(pool)}", flush=True)

    train = known[:N_TRAIN]
    scr_ans = known[N_TRAIN:N_TRAIN + N_SCREEN]
    ev_ans = known[N_TRAIN + N_SCREEN:N_TRAIN + N_SCREEN + N_EVAL_ANS]
    scr_un = unans[:N_SCREEN]
    ev_un = unans[N_SCREEN:N_SCREEN + N_EVAL_UNANS]

    # assert vs abstain contrast rows (for direction + moments)
    rows = [{"question": e["question"], "matching": " " + (e["answer"][0] if e["answer"] else "It is X"),
             "not_matching": " I don't know."} for e in train]

    print("extracting direction + moments ...", flush=True)
    directions = extract_direction(model, tok, rows, "matching", eot=eot)
    mu_a = collect_span_input_moments(model, tok, rows, "matching", all_layers, COMPONENTS, eot=eot)
    mu_b = collect_span_input_moments(model, tok, rows, "not_matching", all_layers, COMPONENTS, eot=eot)

    # baselines on screen/eval
    b_un = abst_rate(gen(model, tok, [e["question"] for e in ev_un], wrap))
    b_an_acc = accuracy(gen(model, tok, [e["question"] for e in ev_ans], wrap),
                        [e["answer"] for e in ev_ans])
    bs_un = abst_rate(gen(model, tok, [e["question"] for e in scr_un], wrap))
    bs_an = abst_rate(gen(model, tok, [e["question"] for e in scr_ans], wrap))
    print(f"base ppl {base_ppl:.2f} | base abstain(unans) {b_un:.3f} | "
          f"base acc(ans) {b_an_acc:.3f}", flush=True)

    # ELS: per-layer selective abstention screen
    print("ELS per-layer screen ...", flush=True)
    q_scr_un = [e["question"] for e in scr_un]
    q_scr_an = [e["question"] for e in scr_ans]
    per = []
    for l in all_layers:
        sc = score_edges(model, directions, mu_a, mu_b, [l], COMPONENTS)
        rk = rank_weight_indices(sc, max(SCREEN_FRAC, 0.01))
        sel = selection_from_ranking(rk, SCREEN_FRAC)
        with pruned_weights(model, sel):
            d_un = abst_rate(gen(model, tok, q_scr_un, wrap)) - bs_un
            d_an = abst_rate(gen(model, tok, q_scr_an, wrap)) - bs_an
            ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
        net = d_un - d_an
        dppl = (ppl - base_ppl) / base_ppl
        per.append({"layer": l, "d_unans": d_un, "d_ans": d_an, "net": net,
                    "d_ppl": dppl, "eff": net / max(dppl, EPS)})
    for r in sorted(per, key=lambda r: -r["net"])[:8]:
        mark = "*" if (r["net"] >= DELTA and r["d_ppl"] <= BETA) else " "
        print(f" {mark} L{r['layer']:>2} net {r['net']:+.3f} (unans {r['d_unans']:+.2f} "
              f"ans {r['d_ans']:+.2f}) Δppl {r['d_ppl']:+.1%}", flush=True)
    passing = [r for r in per if r["net"] >= DELTA and r["d_ppl"] <= BETA]
    L_star = sorted(r["layer"] for r in sorted(passing, key=lambda r: -r["eff"])[:5])
    print(f"L* = {L_star if L_star else 'NONE'}", flush=True)

    report = {"model": MODEL_ID, "env": env_info(), "base_ppl": base_ppl,
              "base_abstain_unans": b_un, "base_acc_ans": b_an_acc,
              "n_answerable_known": len(known), "per_layer": per, "L_star": L_star}

    if L_star:
        q_un = [e["question"] for e in ev_un]
        q_an = [e["question"] for e in ev_ans]
        g_an = [e["answer"] for e in ev_ans]
        scores = score_edges(model, directions, mu_a, mu_b, L_star, COMPONENTS)
        rk = rank_weight_indices(scores, 0.06)
        rk_rnd = rank_weight_indices(random_scores_like(scores, 0), 0.06)
        sweep = []
        for frac in SPARS:
            sel = selection_from_ranking(rk, frac)
            with pruned_weights(model, sel):
                au = abst_rate(gen(model, tok, q_un, wrap))
                oa = gen(model, tok, q_an, wrap)
                acc = accuracy(oa, g_an)
                aa = abst_rate(oa)
                ppl = teacher_forced_ppl(model, tok, wiki, max_tokens=PPL_TOKENS)
            selr = selection_from_ranking(rk_rnd, frac)
            with pruned_weights(model, selr):
                au_r = abst_rate(gen(model, tok, q_un, wrap))
                acc_r = accuracy(gen(model, tok, q_an, wrap), g_an)
            sweep.append({"sparsity": frac, "abstain_unans": au, "acc_ans": acc,
                          "wrong_abstain_ans": aa, "ppl_delta": (ppl - base_ppl) / base_ppl,
                          "rand_abstain_unans": au_r, "rand_acc_ans": acc_r})
            print(f"  s={frac:.3%} abstain_unans {au:.3f} acc_ans {acc:.3f} "
                  f"(rand {au_r:.3f}/{acc_r:.3f}) Δppl {(ppl-base_ppl)/base_ppl:+.1%}",
                  flush=True)
        report["sweep"] = sweep

    RESULTS.mkdir(exist_ok=True)
    tag = MODEL_ID.split("/")[-1].replace(".", "").lower()
    (RESULTS / f"blade_abstention_{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"saved results/blade_abstention_{tag}.json", flush=True)
    if L_star:
        plot(report, tag)


def plot(report, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scienceplots  # noqa
    except Exception as e:
        print("skip plot:", e); return
    b = report
    sw = b["sweep"]
    plt.rcParams["font.family"] = "serif"
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        plt.rc("font", size=13)
        # tradeoff: x = acc on answerable (retained), y = abstain on unanswerable
        ax.plot([b["base_acc_ans"]], [b["base_abstain_unans"]], "*", ms=16,
                color="#333", label="baseline", zorder=5, clip_on=False)
        ax.plot([r["acc_ans"] for r in sw], [r["abstain_unans"] for r in sw], "o-",
                color="#0072B2", lw=2, ms=7, label="BLADE prune", zorder=4, clip_on=False)
        ax.plot([r["rand_acc_ans"] for r in sw], [r["rand_abstain_unans"] for r in sw],
                "s--", color="#D55E00", lw=1.8, ms=6, label="random prune", zorder=3, clip_on=False)
        ax.set_xlabel("accuracy on ANSWERABLE-known (helpfulness kept →)", fontsize=13)
        ax.set_ylabel("abstention on UNANSWERABLE (↑ hallucination avoided)", fontsize=13)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=11, loc="lower left")
        ax.annotate("better", xy=(0.12, 0.9), fontsize=11, color="#0072B2",
                    ha="center", style="italic")
        FIGS.mkdir(exist_ok=True)
        fig.savefig(FIGS / f"blade_abstention_{tag}.pdf", bbox_inches="tight")
        fig.savefig(RESULTS / f"blade_abstention_{tag}.png", dpi=300, bbox_inches="tight")
        print(f"saved figures/blade_abstention_{tag}.pdf", flush=True)


if __name__ == "__main__":
    main()
