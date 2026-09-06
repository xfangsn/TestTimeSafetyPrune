"""BLADE — Behavioral Localization via Activation-Difference Edges.

Generalized behavior-localization pipeline for the Behavior Atlas (applies the
BLADE / signed-actdiff-edge score, defined for refusal in weight_prune.py, to
arbitrary RLHF behaviors).

Any Anthropic model-written eval (A/B format: question + answer_matching_behavior
+ answer_not_matching_behavior) plugs in uniformly. For each behavior we:
  * measure the base model's MC pick-rate (which side it is biased toward),
  * orient the behavior direction toward the model's own bias,
  * extract a CAA direction r (answer-span contrast),
  * score signed-actdiff edges  s = max(r . W . (mu^beh - mu^other), 0),
    where mu is the mean writer INPUT over the behavior-side vs other answer spans,
  * (driver) sweep sparsity to find how concentrated the behavior is.

This is deliberately the SAME recipe as the refusal edge experiment, so the
selected weight sets live in the same index space and are directly comparable.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import torch

from .hooks import capture_span_mean, get_decoder_layers
from .models import chat_wrap
from .sycophancy import _answer_logprob, score_edges  # reuse
from .weight_edit import iter_residual_writers
from .weight_prune import (pruned_weights, rank_weight_indices,
                           selection_from_ranking)

SEED = 0
AAIR = ("https://raw.githubusercontent.com/anthropics/evals/main/"
        "advanced-ai-risk/human_generated_evals/")
PERSONA = "https://raw.githubusercontent.com/anthropics/evals/main/persona/"
SYCO = ("https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/"
        "sycophancy_on_nlp_survey.jsonl")

# behavior name -> dataset url. Refusal is handled separately (existing edge.pt).
CATALOG = {
    "sycophancy": SYCO,
    "corrigibility": AAIR + "corrigible-neutral-HHH.jsonl",
    "power-seeking": AAIR + "power-seeking-inclination.jsonl",
    "survival-instinct": AAIR + "survival-instinct.jsonl",
    "self-awareness": AAIR + "self-awareness-general-ai.jsonl",
    "wealth-seeking": AAIR + "wealth-seeking-inclination.jsonl",
    "myopic-reward": AAIR + "myopic-reward.jsonl",
    # deception / evaluation-gaming (persona collection, Yes/No format)
    "deception": PERSONA + "okay-with-deceiving-a-user-to-help-them.jsonl",
    "manipulate-overseers": PERSONA + "willingness-to-manipulate-overseers-to-think-it-is-HHH.jsonl",
    "self-rate-highly": PERSONA + "willingness-to-rate-own-statements-highly-to-look-better.jsonl",
    "rated-HHH-over-being-HHH": PERSONA + "desire-for-being-rated-HHH-over-actually-being-HHH.jsonl",
    # social-bias probe: BBQ ambiguous+neg questions cast as "stereotyped answer
    # vs 'unknown'"; matching_behavior = biased answer. Pre-built local cache
    # (data/behaviors/bias-bbq.jsonl); URL is a placeholder (cache always exists).
    "bias-bbq": "local://bias-bbq",
    # "evil"/malevolence: dark-triad (psychopathy/machiavellianism/narcissism) +
    # against-human-values persona items, composited. Pre-built local cache
    # (data/behaviors/evil.jsonl). Matches Persona Vectors' "evil" trait in spirit.
    "evil": "local://evil",
}

SYCO_SPLIT_MARK = "Do you agree or disagree with the following claim"


def fetch_ab(name: str, cache_dir: Path) -> list[dict]:
    """Download + parse an A/B eval into {question, matching, not_matching}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.jsonl"
    url = CATALOG[name]
    if not cache.exists():
        with urllib.request.urlopen(url, timeout=120) as resp:
            cache.write_bytes(resp.read())
    rows = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append({
            "question": d["question"],
            "matching": d["answer_matching_behavior"].strip(),
            "not_matching": d["answer_not_matching_behavior"].strip(),
        })
    return rows


def make_splits(rows, n_train=300, n_val=150, n_test=150):
    """50/25/25 split capped at (300,150,150); scales down for small datasets."""
    import random
    idx = list(range(len(rows)))
    random.Random(SEED).shuffle(idx)
    total = min(len(rows), n_train + n_val + n_test)
    nt = min(n_train, total // 2)
    nv = min(n_val, (total - nt) // 2)
    sel = [rows[i] for i in idx[:total]]
    return {"train": sel[:nt], "val": sel[nt:nt + nv], "test": sel[nt + nv:]}


@torch.no_grad()
def pick_rate(model, tokenizer, rows, side_key="matching", batch_size=16):
    """Fraction of items where P(side answer) > P(other answer)."""
    other = "not_matching" if side_key == "matching" else "matching"
    prompts = [chat_wrap(tokenizer, r["question"]) for r in rows]
    a = _answer_logprob(model, tokenizer, prompts, [r[side_key] for r in rows], batch_size)
    b = _answer_logprob(model, tokenizer, prompts, [r[other] for r in rows], batch_size)
    return (a > b).float().mean().item(), (a - b).mean().item()


@torch.no_grad()
def extract_direction(model, tokenizer, rows, side_key, batch_size=16,
                      eot="<|eot_id|>"):
    """r_l = mean act(behavior-side span) - mean act(other span), per layer."""
    other = "not_matching" if side_key == "matching" else "matching"
    prompts = [chat_wrap(tokenizer, r["question"]) for r in rows]
    beh = capture_span_mean(model, tokenizer, prompts, [r[side_key] for r in rows],
                            batch_size=batch_size, eot=eot)
    oth = capture_span_mean(model, tokenizer, prompts, [r[other] for r in rows],
                            batch_size=batch_size, eot=eot)
    return {l: beh[l].mean(0) - oth[l].mean(0) for l in beh}


@torch.no_grad()
def collect_span_input_moments(model, tokenizer, rows, side_key, layers, components,
                               batch_size=16, eot="<|eot_id|>"):
    """Mean writer-INPUT activation over the given answer span, per writer matrix."""
    writers = dict(iter_residual_writers(model, layers, components))
    sums = {n: torch.zeros(m.in_features, device=model.device, dtype=torch.float32)
            for n, m in writers.items()}
    total = 0
    state = {}

    def make_hook(name):
        def hook(_m, args):
            v = args[0].float()  # (B, T, in)
            acc = 0.0
            cnt = 0
            picked = []
            for b, (s, e) in enumerate(state["spans"]):
                picked.append(v[b, s:e].sum(0))
                cnt += (e - s)
            state["b"][name] = (torch.stack(picked).sum(0), cnt)
        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in writers.items()]
    prompts = [chat_wrap(tokenizer, r["question"]) for r in rows]
    texts = [p + r[side_key] + eot for p, r in zip(prompts, rows)]
    # response spans via common-prefix (mirrors capture_span_mean)
    spans_all = []
    for p, t in zip(prompts, texts):
        pid = tokenizer(p, add_special_tokens=False)["input_ids"]
        fid = tokenizer(t, add_special_tokens=False)["input_ids"]
        m = 0
        for x, y in zip(pid, fid):
            if x != y:
                break
            m += 1
        spans_all.append((m, len(fid)))
    try:
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            padding_side="right", add_special_tokens=False
                            ).to(model.device)
            state["spans"] = spans_all[s:s + batch_size]
            state["b"] = {}
            model(**enc, use_cache=False)
            for n in writers:
                vsum, cnt = state["b"][n]
                sums[n] += vsum
            total += sum(e - s2 for s2, e in spans_all[s:s + batch_size])
    finally:
        for h in handles:
            h.remove()
    return {n: (v / total).cpu() for n, v in sums.items()}


def behavior_edge_scores(model, tokenizer, train_rows, side_key, layers, components,
                         eot="<|eot_id|>"):
    """Full edge-score computation for one behavior (direction + span-input mu)."""
    directions = extract_direction(model, tokenizer, train_rows, side_key, eot=eot)
    other = "not_matching" if side_key == "matching" else "matching"
    mu_beh = collect_span_input_moments(model, tokenizer, train_rows, side_key,
                                        layers, components, eot=eot)
    mu_oth = collect_span_input_moments(model, tokenizer, train_rows, other,
                                        layers, components, eot=eot)
    return score_edges(model, directions, mu_beh, mu_oth, layers, components), directions


# --------------------------------------------------------------------------- #
# Effective-Layer Selection (ELS) — data-driven multi-layer choice
# --------------------------------------------------------------------------- #
def solo_layer_pool(model, directions, mu_a, mu_b, all_layers, components,
                    ppl_fn, base_ppl, screen_frac=0.005, beta=0.05, score_fn=None,
                    ranking_fn=None):
    """Candidate pool for ELS: layers whose SOLO prune stays within the ppl
    budget. This drops capability-critical layers (e.g. L0) so best-first never
    wastes budget on them; it is a diagnostic/filter, NOT the selection itself.
    `ppl_fn()` returns current wikitext ppl (called inside the prune context).
    `score_fn(model, directions, mu_a, mu_b, layers, components)` defaults to score_edges;
    pass a BLADE-G scorer to select generic-importance-aware weights.
    Optional ranking_fn(model, directions, mu_a, mu_b, layers, components,
    max_fraction) replaces scoring/ranking, e.g. a fit-local ELS cache.rank."""
    score_fn = score_fn or score_edges
    pool = []
    for l in all_layers:
        if ranking_fn is None:
            sc = score_fn(model, directions, mu_a, mu_b, [l], components)
            ranking = rank_weight_indices(sc, max(screen_frac, 0.01))
        else:
            ranking = ranking_fn(model, directions, mu_a, mu_b, [l],
                                 components, max(screen_frac, 0.01))
        sel = selection_from_ranking(ranking, screen_frac)
        with pruned_weights(model, sel):
            ppl = ppl_fn()
        if (ppl - base_ppl) / base_ppl <= beta:
            pool.append(l)
    return pool


def bestfirst_layers(model, directions, mu_a, mu_b, pool, components,
                     measure, base_metric, base_ppl, beta=0.05, eps=0.005,
                     test_frac=0.005, score_fn=None, ranking_fn=None,
                     bounded_measure=None):
    """ELS selection step: best-first greedy JOINT layer selection.

    Repeatedly add the single layer that most reduces the joint behavior metric
    (measured by pruning L*∪{l} at `test_frac`), subject to ppl<=beta, until no
    layer improves the joint metric by >= eps. Order-independent and captures
    synergy (layers useful only in combination), with no top-k cap / fixed
    order / hard margin — only beta (ppl budget) and eps (stop threshold).
    `measure()` is called inside each prune context and returns (metric, ppl);
    lower metric = more removed. Returns the ordered list of selected layers.
    Optional ranking_fn has the same contract as in solo_layer_pool; it is
    called outside pruning contexts on restored weights.
    Optional bounded_measure(best_metric) replaces measure() inside the prune
    context. Return None only when a rigorous lower bound proves the candidate
    cannot strictly improve this round's best metric; otherwise return the
    fully evaluated (metric, ppl). Bounds must not use eps or partial rates."""
    score_fn = score_fn or score_edges
    selected, current = [], base_metric
    while True:
        best_l, best_m = None, current
        for l in pool:
            if l in selected:
                continue
            cand = sorted(selected + [l])
            if ranking_fn is None:
                sc = score_fn(model, directions, mu_a, mu_b, cand, components)
                ranking = rank_weight_indices(sc, max(test_frac, 0.01))
            else:
                ranking = ranking_fn(model, directions, mu_a, mu_b, cand,
                                     components, max(test_frac, 0.01))
            sel = selection_from_ranking(ranking, test_frac)
            with pruned_weights(model, sel):
                result = (measure() if bounded_measure is None
                          else bounded_measure(best_m))
            if result is None and bounded_measure is not None:
                continue
            m, ppl = result
            if (ppl - base_ppl) / base_ppl <= beta and m < best_m:
                best_l, best_m = l, m
        if best_l is not None and best_m < current - eps:
            selected.append(best_l)
            current = best_m
        else:
            break
    return selected
