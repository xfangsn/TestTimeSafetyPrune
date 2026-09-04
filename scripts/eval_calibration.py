"""P0 calibration eval for the uncertainty pattern: does editing (BLADE-G remove) hurt the model's
ability to know when it is right? Per (condition x benchmark): greedy answer + verbalized confidence,
scored, then Brier / ECE / AUROC / risk-coverage of confidence-vs-correctness + accuracy + lengths.
One condition x bench per run (model-resident) so it shards across Slurm jobs.

Benchmarks load from offline files ($TTS_GSM8K_FILE / $TTS_MATH_FILE), jsonl {question, answer}.
Modes: base | remove | random | shuffle  (all vs the SAME uncertainty ELS layers; BLADE-G edit).
Usage: --model Qwen/Qwen3-14B --dirs qwen3_14b_dirs.pt --els reasoning_els_qwen3_14b.json
       --bench math --mode remove --rho 0.008 --n 200 --out calib_qwen3_14b_math_remove.json
"""
import argparse
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import torch

from ttsafety.models import load_model
from ttsafety.hooks import get_decoder_layers
from ttsafety.weight_prune import pruned_weights, rank_weight_indices, selection_from_ranking, random_scores_like
from ttsafety.sycophancy import score_edges
from reasoning_mask import build_mask, collect_Q

RESULTS = Path("results"); MAX_NEW = 4096


def load_bench(bench, n):
    env = {"gsm8k": "TTS_GSM8K_FILE", "math": "TTS_MATH_FILE"}[bench]
    p = os.environ.get(env)
    rows = [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows[:n]


def qwen_wrap(tok, instr, think=True):
    m = [{"role": "user", "content": instr}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen(model, tok, prompts, max_new, bs=8, do_sample=False):
    prev = tok.padding_side; tok.padding_side = "left"; outs = []
    try:
        for s in range(0, len(prompts), bs):
            enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=do_sample,
                               pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return outs


def last_boxed(s):
    """Extract the content of the LAST \\boxed{...} with balanced braces (handles \\frac{}{} etc.)."""
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def norm_math(s):
    s = str(s).strip()
    s = re.sub(r"\\text\{[^{}]*\}", "", s)
    s = re.sub(r"\\mbox\{[^{}]*\}", "", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("\\:", ""), ("\\ ", ""), ("\\$", ""), ("$", ""), ("\\%", ""), ("%", ""),
                 ("^{\\circ}", ""), ("^\\circ", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\cdot", "*"), ("\\times", "*"), ("\\pi", "pi"), ("{", ""), ("}", ""), (" ", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\\frac(\d)(\d)", r"(\1)/(\2)", s)      # \frac12 -> (1)/(2)
    s = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s)
    s = s.replace("\\sqrt", "sqrt").replace("\\", "")
    s = s.rstrip(".").replace(",", "")
    return s


def extract_answer(text, bench):
    body = text.split("</think>")[-1] if "</think>" in text else text
    if bench == "gsm8k":
        m = re.findall(r"-?\d[\d,]*\.?\d*", body.replace(",", ""))
        return m[-1] if m else ""
    b = last_boxed(body)
    if b is not None:
        return b
    m = re.findall(r"-?\d+\.?\d*", body)
    return m[-1] if m else ""


def _sympy_eq(a, b):
    try:
        import sympy as sp
        A = sp.sympify(a.replace("^", "**")); B = sp.sympify(b.replace("^", "**"))
        return bool(sp.simplify(A - B) == 0)
    except Exception:
        return False


def correct(pred, gold, bench):
    if pred == "" or pred is None:
        return 0
    if bench == "gsm8k":
        try:
            return int(abs(float(str(pred).replace(",", "")) - float(str(gold).replace(",", ""))) < 1e-4)
        except Exception:
            return int(str(pred).strip() == str(gold).strip())
    a, b = norm_math(pred), norm_math(gold)
    if a == b:
        return 1
    try:
        if abs(float(a) - float(b)) < 1e-6:
            return 1
    except Exception:
        pass
    return int(_sympy_eq(a, b))


# ---- calibration metrics ----
def brier(conf, corr):
    return sum((c - y) ** 2 for c, y in zip(conf, corr)) / len(conf)


def ece(conf, corr, bins=10):
    tot = len(conf); e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (c > lo or (b == 0 and c == 0)) and c <= hi]
        if not idx:
            continue
        acc = sum(corr[i] for i in idx) / len(idx); cf = sum(conf[i] for i in idx) / len(idx)
        e += len(idx) / tot * abs(acc - cf)
    return e


def auroc(conf, corr):
    pos = [c for c, y in zip(conf, corr) if y == 1]; neg = [c for c, y in zip(conf, corr) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def risk_coverage_auc(conf, corr):
    order = sorted(range(len(conf)), key=lambda i: -conf[i]); risks = []; err = 0
    for k, i in enumerate(order, 1):
        err += (1 - corr[i]); risks.append(err / k)
    return sum(risks) / len(risks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--dirs", required=True)
    ap.add_argument("--els", required=True); ap.add_argument("--bench", required=True, choices=["gsm8k", "math"])
    ap.add_argument("--mode", required=True, choices=["base", "remove", "random", "shuffle"])
    ap.add_argument("--behavior", default="uncertainty-estimation")
    ap.add_argument("--rho", type=float, default=0.008); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    D = torch.load(RESULTS / args.dirs, weights_only=False); assert D["model"] == args.model
    dirs, muC, muG = D["dirs"], D["muC"], D["muG"]
    ELS = json.loads((RESULTS / args.els).read_text())["els"]
    L = ELS[args.behavior]["L_star"] or ELS[args.behavior]["pool"][:1]
    rows = load_bench(args.bench, args.n)
    print(f"{args.bench} n={len(rows)} | mode={args.mode} | L={L}", flush=True)

    # build edit context
    if args.mode == "base":
        @contextmanager
        def cm():
            yield
        edit = cm()
    else:
        allL = list(range(len(get_decoder_layers(model))))
        Q, _ = collect_Q(model, tok, L)
        if args.mode == "shuffle":
            g = torch.Generator().manual_seed(123)
            r = {l: dirs[args.behavior][l][torch.randperm(dirs[args.behavior][l].numel(), generator=g)]
                 for l in dirs[args.behavior]}
            sel, _ = build_mask(model, r, muC[args.behavior], muG, L, Q=Q, alpha=0.0, rho=args.rho)
        elif args.mode == "random":
            S = score_edges(model, dirs[args.behavior], muC[args.behavior], muG, L, "both")
            rk = rank_weight_indices(random_scores_like(S, seed=args.seed), max(0.03, args.rho))
            sel = selection_from_ranking(rk, args.rho)
        else:  # remove (BLADE-G)
            sel, _ = build_mask(model, dirs[args.behavior], muC[args.behavior], muG, L, Q=Q, alpha=0.0, rho=args.rho)
        edit = pruned_weights(model, sel)

    with edit:
        ans_prompts = [qwen_wrap(tok, r["question"], think=True) for r in rows]
        gens = gen(model, tok, ans_prompts, MAX_NEW)
        preds = [extract_answer(g, args.bench) for g in gens]
        # verbalized confidence (thinking off, short)
        conf_prompts = [qwen_wrap(tok, f"Question: {r['question']}\n\nProposed answer: {p}\n\n"
                                  "How confident are you (0-100) that this answer is correct? "
                                  "Reply with ONLY an integer 0-100.", think=False)
                        for r, p in zip(rows, preds)]
        confs_raw = gen(model, tok, conf_prompts, 12)

    corr = [correct(p, r["answer"], args.bench) for p, r in zip(preds, rows)]
    def parse_conf(s):
        m = re.findall(r"\d+", s.split("</think>")[-1]);
        return min(1.0, max(0.0, int(m[0]) / 100)) if m else 0.5
    vconf = [parse_conf(c) for c in confs_raw]
    tlen = [len(g.split("</think>")[0].split()) for g in gens]
    noans = sum(p == "" for p in preds)

    out = {"model": args.model, "bench": args.bench, "mode": args.mode, "behavior": args.behavior,
           "L": L, "rho": args.rho, "n": len(rows),
           "accuracy": sum(corr) / len(corr), "no_answer": noans / len(rows),
           "think_words": sum(tlen) / len(tlen),
           "verbalized": {"brier": brier(vconf, corr), "ece": ece(vconf, corr),
                          "auroc": auroc(vconf, corr), "risk_cov_auc": risk_coverage_auc(vconf, corr)},
           "items": [{"q": r["question"][:200], "pred": p, "gold": r["answer"], "correct": c,
                      "vconf": vc, "tlen": tl}
                     for r, p, c, vc, tl in zip(rows, preds, corr, vconf, tlen)]}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    v = out["verbalized"]
    print(f"acc {out['accuracy']:.3f} noans {out['no_answer']:.2f} | Brier {v['brier']:.3f} "
          f"ECE {v['ece']:.3f} AUROC {v['auroc']:.3f} RC-AUC {v['risk_cov_auc']:.3f} | saved {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
